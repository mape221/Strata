# Integración opcional con opencode

Estos plugins acompañan a la caché SSD por ID de Strata y a la estabilización del prefijo build/plan. Funcionan con el proveedor cuyo `providerID` es `strata` y se probaron con opencode 1.18.35.

- `diag-session.ts`: instalar como `<proyecto>/.opencode/plugins/diag-session.ts`. Envía `X-Session-Id` con el ID de la sesión de opencode. Registra eventos en `<proyecto>/.opencode/plugins/diag-session.log`.
- `stable-strata-reminders.js`: instalar como `~/.config/opencode/plugins/stable-strata-reminders.js` (o en los plugins del proyecto). Repone los recordatorios sintéticos de turnos históricos de plan/build que opencode deja fuera al reconstruir el historial; no cambia los mensajes almacenados.

No es necesario instalar dos copias del mismo plugin. Tras copiar o actualizar un plugin, cerrar y volver a abrir opencode para que cargue el cambio. Si se actualiza opencode, contrastar los textos de recordatorio de este plugin con los de la nueva versión.

La estabilización de la descripción de `task` se realiza **en `serve/server.py`**, no en estos plugins. Los permisos reales de plan siguen bajo el control de opencode.
