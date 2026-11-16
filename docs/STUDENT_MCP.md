# Student data tools (MCP)

Lets the chatbot answer personal questions: timetable, invoices, attendance. Read-only.

## Flow
Spring verifies the JWT, then sends `studentEmail` in `POST /internal/chat`. The agent's Reason stage
classifies the question as `STUDENT_DATA`; `StudentDataStage` lets the LLM pick tools, runs them for that
email, and the LLM writes the reply from the results.

## Safety rules
- The email is **never** a tool argument; it comes from the verified request.
- Invoice payment links, slip URLs and line items are stripped before the LLM sees them.
- Tool output is treated as data (the answer prompt ignores instructions inside it).
- Spring down -> friendly message; no `studentEmail` -> "can't look that up".

## Files
- `app/domain/ports/student_data.py` - port. `app/infrastructure/clients/spring_student_data.py` - HTTP adapter.
- `app/services/student_tools.py` - the tools (single source for agent and MCP).
- `app/services/agent/student_data.py` - agent stage. `app/mcp_server/server.py` - MCP server.

## Spring contract (to be built in the backend)
`GET {SPRING_BASE_URL}/internal/student/{timetable|invoices|attendance}?studentEmail=...` with header
`X-Internal-Secret`. Each returns a JSON array: timetable = `CalenderTimeSlotDto`, invoices = `Invoice`,
attendance = `StudentCourseStatsDto`.

## Try the MCP server
`set MCP_STUDENT_EMAIL=student@example.com` then `npx @modelcontextprotocol/inspector python -m app.mcp_server.server`
