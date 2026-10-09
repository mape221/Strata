// CPU-only tests for the spill directory (include/strata/core/conversation_spill.hpp): a parked conversation is
// written as a session file plus a metadata sidecar, matched from the sidecar alone, and read back whole.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_spill.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}
std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}
ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.gdn = bytes_of(4099, seed); c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2); c.dead = bytes_of(64, seed + 3); c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}
SavedConversation sample(size_t tokens) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48; s.cvec = false;
    s.live = checkpoint(tokens, 1);
    s.checkpoints.push_back(checkpoint(tokens / 4, 2));
    s.checkpoints.push_back(checkpoint(tokens / 2, 3));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = 9; kv.idx_dim = 128;
        kv.k = pattern(4096, uint8_t(layer)); kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(512, uint8_t(layer + 20)); kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}
bool buffers_equal(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    bool same = true;
    size_t at = 0;
    a.visit(0, a.size(), [&](const uint8_t* p, size_t c, size_t off) {
        b.visit(off, c, [&](const uint8_t* q, size_t c2, size_t) {
            if (c2 != c) { same = false; return false; }
            for (size_t i = 0; i < c; ++i) if (p[i] != q[i]) { same = false; return false; }
            return true;
        });
        at += c;
        return same;
    });
    return same && at == a.size();
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    ConversationSpillCache spill;
    std::string error;
    check(spill.open(dir, id, 1ull << 30, error), "open empty dir");
    check(spill.enabled(), "enabled");
    check(spill.size() == 0, "empty index");
    check(spill.miss_reason(std::vector<int32_t>{1, 2}, {}, false, 0.0, 64).find("no indexed SSD") !=
          std::string::npos, "miss: empty disk tier");

    std::vector<ConversationCheckpoint> branch_points;
    for (size_t n : {100u, 1000u, 2200u, 4000u, 6500u, 7000u}) branch_points.push_back(checkpoint(n, 1));
    check(conversation_spill_anchor(branch_points, 5) == 2,
          "sparse checkpoint keeps an older useful branch near one third of the tip");
    branch_points[1].pinned = true;
    check(conversation_spill_anchor(branch_points, 5) == 1,
          "sparse checkpoint preserves an explicit pinned branch");
    check(conversation_spill_anchor(branch_points, 999) == branch_points.size(),
          "sparse checkpoint refuses an invalid tip");

    SavedConversation a = sample(1000);
    check(spill.spill(a, error), "spill first");
    check(spill.size() == 1, "one entry");
    // the prompt continues the conversation (one token past its end): the whole live state resumes.
    std::vector<int32_t> continuing = a.live.ids;
    continuing.push_back(int32_t(7777));
    const auto first = spill.best(continuing, a.live.imgs, false, 0.0, 0);
    check(bool(first), "match own conversation");
    check(first.tokens == (int64_t) a.live.ids.size(), "full prefix match");
    check(first.live, "live match");

    // read it back whole and compare the K/V byte for byte
    SavedConversation back;
    check(spill.load(first.path, back, {}, error), "load back");
    check(back.live.ids == a.live.ids, "ids round-trip");
    check(back.checkpoints.size() == a.checkpoints.size(), "checkpoint count round-trip");
    check(back.kv.size() == a.kv.size(), "kv layer count");
    bool kv_same = true;
    for (size_t i = 0; i < back.kv.size(); ++i)
        kv_same = kv_same && buffers_equal(back.kv[i].k, a.kv[i].k) && buffers_equal(back.kv[i].v, a.kv[i].v) &&
                  buffers_equal(back.kv[i].pooled, a.kv[i].pooled);
    check(kv_same, "kv bytes round-trip");

    // a different prompt sharing a prefix resumes at the common prefix, not the whole thing
    std::vector<int32_t> other = a.live.ids;
    other.resize(400);
    for (size_t i = 400; i < a.live.ids.size(); ++i) other.push_back(int32_t(9000 + i));
    check(other.size() == a.live.ids.size(), "other same length, diverges at 400");
    const auto partial = spill.best(other, {}, false, 0.0, 0);
    // the resume lands on the deepest checkpoint still a prefix (250); the 500 checkpoint crosses the divergence.
    check(bool(partial) && partial.tokens == 250, "resume at the deepest still-valid checkpoint");
    check(!partial.live, "checkpoint resume, not live");

    // similarity filters a weak hit: 400/1000 = 0.4 LCP is refused at similarity 0.5
    const auto weak = spill.best(other, {}, false, 0.5, 0);
    check(!bool(weak), "similarity filters weak hit");
    check(spill.miss_reason(other, {}, false, 0.5, 0).find("below similarity threshold") !=
          std::string::npos, "miss: similarity threshold");
    // n_min filters a short hit
    const auto short_hit = spill.best(std::vector<int32_t>(a.live.ids.begin(), a.live.ids.begin() + 10), {}, false, 0.0, 100);
    check(!bool(short_hit), "n_min filters short hit");
    check(spill.miss_reason(other, {}, false, 0.0, 500).find("below minimum shared prefix 500") !=
          std::string::npos, "miss: minimum shared prefix");

    // A 68k-token conversation with only a late checkpoint has the same failure mode: matching some history
    // is insufficient when no stored running state precedes the point where the client history diverges.
    std::vector<int32_t> before_checkpoint = a.live.ids;
    for (size_t i = 100; i < before_checkpoint.size(); ++i) before_checkpoint[i] += 12345;
    check(!bool(spill.best(before_checkpoint, {}, false, 0.0, 0)), "divergence before first checkpoint misses");
    const std::string checkpoint_miss = spill.miss_reason(before_checkpoint, {}, false, 0.0, 0);
    check(checkpoint_miss.find("index 100") != std::string::npos &&
          checkpoint_miss.find("earliest checkpoint 250") != std::string::npos,
          "miss: exact divergence and first restorable checkpoint");
    std::vector<int32_t> unrelated = a.live.ids;
    unrelated[0] += 12345;
    check(spill.miss_reason(unrelated, {}, false, 0.0, 0).find("no shared token/image prefix") !=
          std::string::npos, "miss: entirely unrelated prompt");

    // cvec mismatch never matches
    const auto wrong_cvec = spill.best(a.live.ids, a.live.imgs, true, 0.0, 0);
    check(!bool(wrong_cvec), "cvec mismatch refused");
    check(spill.miss_reason(a.live.ids, {}, true, 0.0, 0).find("cvec mode") != std::string::npos,
          "miss: cvec mode");

    fs::remove_all(dir);
    ConversationSpillCache with_images;
    check(with_images.open(dir, id, 1ull << 30, error), "open for image mismatch test");
    SavedConversation image = sample(1000);
    image.live.imgs.push_back({100, 456});
    for (auto& ck : image.checkpoints) ck.imgs.push_back({100, 456});
    check(with_images.spill(image, error), "save image fixture");
    const std::vector<ConversationImageKey> changed_image{{100, 789}};
    check(!bool(with_images.best(continuing, changed_image, false, 0.0, 0)),
          "different image identity misses");
    const std::string image_miss = with_images.miss_reason(continuing, changed_image, false, 0.0, 0);
    check(image_miss.find("index 100") != std::string::npos &&
          image_miss.find("image identity differs") != std::string::npos,
          "miss: image identity and boundary, without content");

    // reopen the directory: the sidecar is reindexed without reading the session file
    fs::remove_all(dir);
    ConversationSpillCache ordinary;
    check(ordinary.open(dir, id, 1ull << 30, error), "recreate ordinary fixture directory");
    check(ordinary.spill(a, error), "recreate ordinary fixture");
    ConversationSpillCache again;
    check(again.open(dir, id, 1ull << 30, error), "reopen dir");
    check(again.size() == 1, "reindexed one entry");
    const auto rematched = again.best(continuing, a.live.imgs, false, 0.0, 0);
    check(bool(rematched) && rematched.tokens == (int64_t) a.live.ids.size(), "rematch after reopen");

    // a foreign identity is refused and its files removed
    ConversationSpillCache foreign;
    check(foreign.open(dir, SessionFileIdentity{1, 2}, 1ull << 30, error), "open with foreign identity");
    check(foreign.size() == 0, "foreign identity: nothing indexed");
    check(fs::exists(dir / "strata-conv-1.sess"), "another identity's files are left alone");

    // budget eviction drops the oldest
    ConversationSpillCache tight;
    check(tight.open(dir, id, 1ull << 30, error), "open for budget test");
    const uint64_t before = tight.bytes();
    SavedConversation b = sample(2000);
    b.live.ids[0] += 7777;
    for (auto& c : b.checkpoints) c.ids[0] += 7777;   // a different conversation: no full-prefix supersession
    check(tight.spill(b, error), "spill second");
    const uint64_t newest = tight.bytes() - before;   // the newest session file's bytes
    // a budget that fits the newest alone: reopening keeps it and evicts the older one.
    ConversationSpillCache small;
    check(small.open(dir, id, newest, error), "open with a budget for one");
    check(small.size() == 1, "budget keeps one");
    check(small.disk_evictions() >= 1, "budget evicted the oldest");

    // the RAM cache handing its evictions to the disk tier: make_room's spill callback writes what it drops.
    fs::remove_all(dir);
    ConversationSpillCache tier;
    check(tier.open(dir, id, 1ull << 30, error), "open for the RAM-cache tier test");
    SavedConversation c1 = sample(1000), c2 = sample(1500);
    c2.live.ids[0] += 7777;
    for (auto& c : c2.checkpoints) c.ids[0] += 7777;
    const size_t need = c1.bytes() + c2.bytes();
    ConversationCache ram(need, 1);   // one slot: putting the second evicts the first
    check(ram.enabled(), "ram cache enabled");
    check(ram.put(std::move(c1)), "put first in RAM");
    size_t spilled = 0;
    auto spill_cb = [&](const SavedConversation& evicted) { ++spilled; tier.spill(evicted, error); };
    // make_room for the second conversation evicts the first and hands it to the callback.
    check(ram.make_room(c2.bytes(), 0, spill_cb), "make_room evicts to make space");
    check(spilled == 1, "one conversation handed to the spill callback");
    check(tier.size() == 1, "the evicted conversation reached disk");
    // the disk copy resumes the same conversation the RAM cache dropped.
    std::vector<int32_t> cont1 = sample(1000).live.ids;
    cont1.push_back(int32_t(7777));
    const auto from_disk = tier.best(cont1, sample(1000).live.imgs, false, 0.0, 0);
    check(bool(from_disk) && from_disk.tokens == 1000, "disk resumes the evicted conversation");
    // spill_all empties the RAM cache onto disk (the shutdown path).
    check(ram.put(std::move(c2)), "put second in RAM");
    const size_t drained = ram.spill_all(spill_cb);
    check(drained == 1, "spill_all drained the parked conversation");
    check(ram.size() == 0, "ram cache empty after spill_all");
    check(tier.size() == 2, "both conversations now on disk");

    // Disk-only mode: a live session is written using source callbacks, with no full K/V image allocated on RAM.
    fs::remove_all(dir);
    ConversationSpillCache streamed;
    check(streamed.open(dir, id, 1ull << 30, error), "open disk-only cache");
    SavedConversation live = sample(2000);
    SavedConversation meta = live;
    meta.kv.clear();
    std::vector<SessionKvSource> sources;
    for (const ConversationKv& kv : live.kv) {
        SessionKvSource src;
        src.format = kv.format; src.cells = kv.cells; src.heads = kv.heads;
        src.head_dim = kv.head_dim; src.page_size = kv.page_size;
        src.pooled_rows = kv.pooled_rows; src.idx_dim = kv.idx_dim;
        const ConversationBuffer* parts[] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        for (size_t p = 0; p < 5; ++p) src.sizes[p] = parts[p]->size();
        src.read = [parts = std::array<const ConversationBuffer*, 5>{parts[0], parts[1], parts[2], parts[3], parts[4]}]
                   (size_t part, size_t offset, void* dst, size_t n) {
            if (part >= parts.size() || offset > parts[part]->size() || n > parts[part]->size() - offset) return false;
            size_t copied = 0;
            return parts[part]->visit(offset, n, [&](const uint8_t* data, size_t count, size_t) {
                std::memcpy(static_cast<uint8_t*>(dst) + copied, data, count);
                copied += count;
                return true;
            });
        };
        sources.push_back(std::move(src));
    }
    check(streamed.spill_sources(meta, sources, 1ull << 20, error), "stream the live K/V to disk");
    const auto disk = streamed.best(std::vector<int32_t>(live.live.ids.begin(), live.live.ids.end()),
                                     live.live.imgs, false, 0.0, 0);
    check(bool(disk), "disk-only session matches without RAM parking");
    check(disk.checkpoints == 2, "sidecar reports the exact number of retained checkpoints");
    SavedConversation read_meta;
    std::vector<std::array<uint64_t, 5>> part_sizes;
    SessionReadLimits too_few;
    too_few.max_checkpoints = 1;
    check(!streamed.load_streamed(disk.path, read_meta, too_few, part_sizes, error),
          "one-checkpoint memory limit refuses a two-checkpoint session");
    read_meta = {};
    part_sizes.clear();
    error.clear();
    check(streamed.load_streamed(disk.path, read_meta, {}, part_sizes, error), "validate a streamed disk read");
    check(read_meta.kv.size() == live.kv.size() && read_meta.kv[0].k.empty(),
          "streamed metadata holds no K/V payload");
    std::vector<std::array<std::vector<uint8_t>, 5>> received(part_sizes.size());
    for (size_t layer = 0; layer < part_sizes.size(); ++layer)
        for (size_t part = 0; part < 5; ++part) received[layer][part].resize((size_t) part_sizes[layer][part]);
    auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
        if (layer >= received.size() || part >= 5 || offset > received[layer][part].size() ||
            n > received[layer][part].size() - offset) return false;
        std::memcpy(received[layer][part].data() + offset, data, n);
        return true;
    };
    check(session_file_apply_streamed(disk.path, id, sink, part_sizes, error), "apply streamed K/V blocks");
    for (size_t layer = 0; layer < live.kv.size(); ++layer) {
        const ConversationBuffer* parts[] = {&live.kv[layer].k, &live.kv[layer].v,
                                             &live.kv[layer].k_scale, &live.kv[layer].v_scale,
                                             &live.kv[layer].pooled};
        for (size_t part = 0; part < 5; ++part) {
            std::vector<uint8_t> want(parts[part]->size());
            parts[part]->visit(0, want.size(), [&](const uint8_t* data, size_t count, size_t at) {
                std::memcpy(want.data() + at, data, count); return true;
            });
            check(received[layer][part] == want, "streamed K/V equals the original bytes");
        }
    }
    SavedConversation newer = sample(2200);
    check(streamed.spill(newer, error), "save a longer version of the same chat");
    check(streamed.size() == 1 && !fs::exists(disk.path), "superseded whole chat removed after new save");
    SavedConversation sibling = sample(2200);
    for (size_t k = 1200; k < sibling.live.ids.size(); ++k) sibling.live.ids[k] += 7777;
    check(streamed.spill(sibling, error), "save a sibling with the same system prompt");
    check(streamed.size() == 2, "common root does not merge distinct conversations");

    fs::remove_all(dir);
    ConversationSpillCache ttl;
    check(ttl.open(dir, id, 1ull << 30, error), "open fresh TTL directory");
    check(ttl.spill(sample(1000), error), "save TTL fixture");
    const auto ttl_path = ttl.best(continuing, {}, false, 0.0, 0).path;
    check(!ttl_path.empty(), "TTL fixture matched");
    ttl.touched(ttl_path);
    ConversationSpillCache fresh;
    check(fresh.open(dir, id, 1ull << 30, error), "recently used disk session reindexed");
    check(fresh.size() == 1, "last access refresh kept a session alive");
    fs::last_write_time(dir / "strata-conv-1.meta",
                        fs::file_time_type::clock::now() - std::chrono::hours(25));
    ConversationSpillCache expired;
    check(expired.open(dir, id, 1ull << 30, error), "expired spill folder reopens");
    check(expired.size() == 0 && !fs::exists(dir / "strata-conv-1.sess"),
          "24-hour expiry removes the session and its metadata");
#ifndef _WIN32
    check((fs::status(dir).permissions() & fs::perms::group_all) == fs::perms::none,
          "spill directory is private");
#endif

    fs::remove_all(dir);
    std::printf("conversation_spill_test: %d checks passed\n", checks);
    return 0;
}
