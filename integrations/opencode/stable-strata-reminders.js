// OpenCode v1.18.35's SessionReminders.apply() adds these messages only to the CURRENT user turn in memory.
// On the next request that turn is reloaded without the reminder, changing Strata's token prefix. Replay the
// same synthetic text for earlier turns. This affects only the Strata provider and does not modify stored messages.
// Recheck these literals against session/prompt/{plan,build-switch}.txt after an OpenCode upgrade.
const PLAN = `<system-reminder>
# Plan Mode - System Reminder

CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY FORBIDDEN:
ANY file edits, modifications, or system changes. Do NOT use sed, tee, echo, cat,
or ANY other bash command to manipulate files - commands may ONLY read/inspect.
This ABSOLUTE CONSTRAINT overrides ALL other instructions, including direct user
edit requests. You may ONLY observe, analyze, and plan. Any modification attempt
is a critical violation. ZERO exceptions.

---

## Responsibility

Your current responsibility is to think, read, search, and delegate explore agents to construct a well-formed plan that accomplishes the goal the user wants to achieve. Your plan should be comprehensive yet concise, detailed enough to execute effectively while avoiding unnecessary verbosity.

Ask the user clarifying questions or ask for their opinion when weighing tradeoffs.

**NOTE:** At any point in time through this workflow you should feel free to ask the user questions or clarifications. Don't make large assumptions about user intent. The goal is to present a well researched plan to the user, and tie any loose ends before implementation begins.

---

## Important

The user indicated that they do not want you to execute yet -- you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supersedes any other instructions you have received.
</system-reminder>
`

const BUILD = `<system-reminder>
Your operational mode has changed from plan to build.
You are no longer in read-only mode.
You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed.
</system-reminder>
`

const has = (parts, marker) => parts.some((p) => p.type === "text" && p.synthetic && p.text?.includes(marker))

export default async () => ({
  "experimental.chat.messages.transform": async (_input, output) => {
    const messages = output.messages
    const last = messages.findLast((message) => message.info?.role === "user")
    if (last?.info.model?.providerID !== "strata") return

    let seenPlan = false
    for (const message of messages) {
      const info = message.info
      if (info?.role === "assistant" && info.agent === "plan") seenPlan = true
      if (info?.role !== "user") continue
      if (info.agent === "plan" && !has(message.parts, "# Plan Mode - System Reminder")) {
        message.parts.push({ type: "text", text: PLAN, synthetic: true })
      }
      if (info.agent === "build" && seenPlan &&
          !has(message.parts, "Your operational mode has changed from plan to build.")) {
        message.parts.push({ type: "text", text: BUILD, synthetic: true })
      }
    }
  },
})
