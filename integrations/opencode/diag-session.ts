import type { Plugin } from "@opencode-ai/plugin"

function sanitizeSessionId(value: unknown): string | null {
  const safe = String(value ?? "")
    .replace(/[^A-Za-z0-9._:-]/g, "-")
    .slice(0, 128)
  return safe.length > 0 ? safe : null
}

export default (async ({ directory }) => {
  const log = `${directory}/.opencode/plugins/diag-session.log`

  function appendLog(text: string): void {
    try {
      const dynamicImport = Function("return import('node:fs')")() as Promise<any>
      dynamicImport.then((fs: any) => fs.appendFile(log, text, () => {})).catch(() => {})
    } catch {}
  }

  return {
    "chat.headers": async (input, output) => {
      const providerID = (input.model && (input.model as any).providerID) ?? "?"
      if (providerID !== "strata") return

      const session = sanitizeSessionId(input.sessionID)
      if (session) output.headers["X-Session-Id"] = session

      const modelID = (input.model && (input.model as any).id) ?? "?"
      const headerKeys = Object.keys(output.headers || {}).join(",")
      appendLog(
        `${new Date().toISOString()} chat.headers agent=${input.agent} sessionID=${input.sessionID} sent=${session ?? "-"} model=${modelID} msgID=${input.message?.id} headers=[${headerKeys}]\n`,
      )
    },
  }
}) satisfies Plugin
