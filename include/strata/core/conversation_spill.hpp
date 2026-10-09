// A durable disk tier for --serve's parked conversations: when the RAM cache evicts a conversation, it is written
// as an ordinary session file (conversation_file.hpp, the same format and model/config identity as the slot save/
// restore API) and can be restored after a restart. A small sidecar beside each file holds only the token/image
// metadata, so a new request finds the best disk match without reading the conversation's K/V.
//
// This is host code only (no CUDA): it calls the session-file writer and reader, which are host code too.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationSpillMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;
    bool live = false;
    size_t checkpoints = 0;      // exact sidecar count, for bounded restore admission
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

// A second disk checkpoint protects an older branch without keeping every running state. Explicitly pinned
// branch points win; otherwise retain the checkpoint nearest one third of the tip (prefer one before it).
inline size_t conversation_spill_anchor(const std::vector<ConversationCheckpoint>& checks, size_t deepest) {
    if (checks.size() <= 1 || deepest >= checks.size()) return checks.size();
    const size_t target = checks[deepest].ids.size() / 3;
    size_t anchor = checks.size();
    for (size_t i = 0; i < checks.size(); ++i) {
        if (i == deepest) continue;
        if (anchor == checks.size() || (checks[i].pinned && !checks[anchor].pinned) ||
            (checks[i].pinned == checks[anchor].pinned &&
             ((checks[i].ids.size() <= target && checks[i].ids.size() > checks[anchor].ids.size()) ||
              (checks[anchor].ids.size() > target && checks[i].ids.size() < checks[anchor].ids.size()))))
            anchor = i;
    }
    return anchor;
}

class ConversationSpillCache {
public:
    // Opens (creating if needed) the spill directory and indexes the conversations already in it. Files whose
    // sidecar names another model/config identity, or whose session file is missing or unreadable, are removed.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              std::string& error, std::chrono::seconds lifetime = std::chrono::hours(24));
    bool enabled() const { return enabled_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t stale_files_wiped() const { return stale_files_wiped_; }
    size_t disk_evictions() const { return disk_evictions_; }

    // The best resume this directory offers for the prompt, from the sidecars only (no K/V read). similarity and
    // n_min filter weak hits exactly as the RAM cache's best() does.
    template<class Token>
    ConversationSpillMatch best(const std::vector<Token>& prompt,
                                const std::vector<ConversationImageKey>& images, bool cvec,
                                double similarity, int64_t n_min) const {
        ConversationSpillMatch best_match;
        int64_t best_tokens = 0;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& entry = entries_[i];
            if (expired(entry)) continue;
            const ConversationMatch match = conversation_metadata_match(entry.live_meta,
                    entry.checkpoint_lengths, prompt, images, cvec, entry.cvec, similarity, n_min);
            if (match.tokens > best_tokens)
                best_tokens = match.tokens, best_match = {entry.session_path(), entry.file_bytes, match.tokens,
                                                          match.live, entry.checkpoint_lengths.size()};
        }
        return best_match;
    }

    // Called only when a request would reread the entire prompt and best() found nothing. Report why the closest
    // indexed session cannot be resumed; token/image contents and IDs never enter the log.
    template<class Token>
    std::string miss_reason(const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images, bool cvec,
                            double similarity, int64_t n_min) const {
        if (entries_.empty()) return "no indexed SSD conversations for this model/config (none saved, expired, or evicted)";
        if (prompt.empty()) return "incoming prompt is empty";
        const Entry* closest = nullptr;
        int64_t shared = -1;
        size_t active = 0;
        for (const Entry& entry : entries_) {
            if (expired(entry)) continue;
            ++active;
            if (entry.cvec != cvec) continue;
            const int64_t lcp = conversation_lcp(entry.live_meta, prompt, images);
            if (lcp > shared) { closest = &entry; shared = lcp; }
        }
        if (!closest) return active == 0 ? "all indexed SSD conversations expired before lookup"
                                         : "no indexed SSD conversation with matching cvec mode";
        const std::string name = closest->stem.filename().string();
        std::string detail = "closest " + name + " (" + std::to_string(closest->live_meta.ids.size()) +
                             " saved tokens): shared prefix " + std::to_string(shared) +
                             "/" + std::to_string(prompt.size()) + " tokens";
        const size_t common = (size_t) shared;
        const size_t limit = std::min(closest->live_meta.ids.size(), prompt.size());
        if (common < limit) {
            detail += "; first incompatible token/image at index " + std::to_string(common) + " (0-based)";
            if (closest->live_meta.ids[common] == prompt[common]) detail += " (image identity differs)";
        } else if (common == prompt.size()) detail += "; incoming prompt ends before a reusable state";
        else detail += "; saved conversation ends before the incoming prompt";
        if (shared == 0) return detail + "; no shared token/image prefix";
        if (shared < n_min) return detail + "; below minimum shared prefix " + std::to_string(n_min);
        if ((double) shared / (double) prompt.size() <= similarity)
            return detail + "; below similarity threshold " + std::to_string(similarity);
        size_t first_checkpoint = 0;
        for (size_t n : closest->checkpoint_lengths)
            if (!first_checkpoint || n < first_checkpoint) first_checkpoint = n;
        detail += "; no restorable state at/before shared prefix (live " +
                  std::to_string(closest->live_meta.ids.size()) + ", earliest checkpoint " +
                  (first_checkpoint ? std::to_string(first_checkpoint) : "none") + ")";
        return detail;
    }

    // Reads a spilled conversation back into image (its K/V included), with the engine's read limits and identity.
    bool load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
              std::string& error) const;
    bool load_streamed(const std::string& path, SavedConversation& meta, const SessionReadLimits& limits,
                       std::vector<std::array<uint64_t, 5>>& sizes, std::string& error) const;
    // Writes a parked conversation (its K/V in RAM) as a session file plus its sidecar, and indexes it.
    bool spill(const SavedConversation& image, std::string& error);
    // A live session's K/V stays on its device: stream the authoritative buffers to disk instead of first
    // allocating a SavedConversation containing the whole K/V in RAM.
    bool spill_sources(const SavedConversation& meta, const std::vector<SessionKvSource>& sources,
                       uint64_t min_free_bytes, std::string& error);
    bool erase(const std::string& path, std::string& error);
    void prune_expired();
    void touched(const std::string& path);
    // A conversation being restored is held out of eviction until it is put back in RAM or dropped.
    void pin(const std::string& path);
    void unpin(const std::string& path);
    // Removes the disk copies of this conversation a turn back, the same rule the RAM cache applies before evicting.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                            const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                            const std::string& keep_path = {});

private:
    struct Entry {
        std::filesystem::path stem;          // "<dir>/strata-conv-<serial>" (both files share it)
        uint64_t file_bytes = 0;
        std::filesystem::file_time_type used_at{};
        bool cvec = true;
        ConversationCheckpoint live_meta;    // ids + imgs only; no running state
        std::vector<size_t> checkpoint_lengths;
        std::string session_path() const { return (stem.string() + ".sess"); }
        std::string meta_path() const { return (stem.string() + ".meta"); }
    };
    void enforce_budget();
    bool expired(const Entry& entry) const;
    std::filesystem::path fresh_stem();
    bool index_written(const std::filesystem::path& stem, const SavedConversation& image,
                       size_t file_bytes, std::string& error);
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity, std::string& error) const;
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error) const;

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    std::chrono::seconds lifetime_{std::chrono::hours(24)};
    size_t stale_files_wiped_ = 0, disk_evictions_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // oldest spill first
    std::string pinned_path_;
};

} // namespace strata::core
