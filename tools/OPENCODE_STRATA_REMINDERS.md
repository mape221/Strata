# Stable OpenCode reminders for Strata

OpenCode 1.18.35 appends plan/build-switch reminders to the current request without saving them in the session. When an earlier turn is replayed after switching conversations, the missing reminder changes the token prefix, preventing Strata from restoring its SSD cache.

Install [`opencode_stable_strata_reminders.js`](opencode_stable_strata_reminders.js) in `~/.config/opencode/plugins/` (or in a project's `.opencode/plugins/`). OpenCode auto-discovers plugins there; do not also list the file in the `plugin` array. Restart OpenCode after copying it. The plugin affects only the `strata` provider and does not modify stored session messages. Recheck the embedded reminder texts against OpenCode's `session/prompt/plan.txt` and `build-switch.txt` when upgrading OpenCode.

SSD session matching still requires an identical token/image prefix. An explicit session ID cannot restore an incompatible K/V state. The engine's optional SSD tier retains an earlier checkpoint when RAM allows it, so a changed branch can resume from an earlier point.
