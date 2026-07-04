-- Rosetta — reconhece contas manuscritas na página e desenha o resultado (estilo iPad).
--
-- Ctrl+M (ou Plugin > Rosetta): envia todos os traços da camada atual para a API local
-- (POST http://localhost:8000/page/process). O servidor acha os "=" pendentes, seleciona a
-- área de cada conta sozinho, reconhece (modelo HMER) e devolve o resultado como traços,
-- que são desenhados logo após o "=". Traços de resultado têm cor própria — é assim que a
-- conta fica marcada como resolvida (rodar de novo não duplica).
--
-- Requisitos: API rodando (scripts/serve_api.ps1 na raiz do repo) e curl.exe (nativo no
-- Windows 10+).

local json = require("json")

local API_URL = "http://localhost:8000/page/process"
local CURL_TIMEOUT_S = 120

function initUi()
  app.registerUi({
    ["menu"] = "Rosetta: reconhecer contas",
    ["callback"] = "rosettaRun",
    ["accelerator"] = "<Control>m",
  })
end

local function showError(msg)
  app.openDialog("Rosetta: " .. msg, { "Ok" }, "", true)
end

local function showInfo(msg)
  app.openDialog("Rosetta: " .. msg, { "Ok" }, "")
end

-- POST do JSON via curl (o Lua do Xournal++ não tem HTTP). Corpo vai em arquivo temporário
-- para não estourar limite de linha de comando nem sofrer com escaping.
local function postJson(url, bodyTable)
  local tmpDir = os.getenv("TEMP") or os.getenv("TMP") or "."
  local reqPath = tmpDir .. "\\rosetta_request.json"

  local f, ferr = io.open(reqPath, "w")
  if not f then
    return nil, "não consegui escrever arquivo temporário: " .. tostring(ferr)
  end
  f:write(json.encode(bodyTable))
  f:close()

  local cmd = string.format(
    'curl -s -m %d -H "Content-Type: application/json" --data-binary "@%s" "%s" 2>nul',
    CURL_TIMEOUT_S, reqPath, url
  )
  local pipe = io.popen(cmd)
  if not pipe then
    return nil, "não consegui executar o curl"
  end
  local resp = pipe:read("*a")
  pipe:close()
  os.remove(reqPath)

  if not resp or resp == "" then
    return nil, "API não respondeu em " .. url .. ".\nSuba o servidor: scripts\\serve_api.ps1"
  end
  local ok, decoded = pcall(json.decode, resp)
  if not ok then
    return nil, "resposta inesperada da API: " .. string.sub(resp, 1, 200)
  end
  return decoded
end

function rosettaRun()
  local ok, strokes = pcall(app.getStrokes, "layer")
  if not ok or strokes == nil or #strokes == 0 then
    showInfo("a camada atual não tem tinta.")
    return
  end

  local payload = { strokes = {} }
  for i, s in ipairs(strokes) do
    payload.strokes[i] = { x = s.x, y = s.y, color = s.color, width = s.width }
  end

  local resp, err = postJson(API_URL, payload)
  if not resp then
    showError(err)
    return
  end
  if resp.detail then -- erro estruturado do FastAPI (ex.: 501 = modelo não carregado)
    showError(type(resp.detail) == "string" and resp.detail or json.encode(resp.detail))
    return
  end

  local newStrokes = {}
  local failures = {}
  for _, expr in ipairs(resp.expressions or {}) do
    if expr.strokes and #expr.strokes > 0 then
      for _, s in ipairs(expr.strokes) do
        table.insert(newStrokes, {
          x = s.x,
          y = s.y,
          tool = "pen",
          width = s.width,
          color = s.color,
        })
      end
    elseif expr.error then
      table.insert(failures, (expr.latex or "?") .. " -> " .. expr.error)
    end
  end

  if #newStrokes > 0 then
    app.addStrokes({ strokes = newStrokes, allowUndoRedoAction = "grouped" })
    app.refreshPage()
  end

  if #newStrokes == 0 and #failures == 0 then
    showInfo("nenhuma conta nova encontrada (escreva a conta terminando em \"=\").")
  elseif #failures > 0 then
    showError("algumas contas não puderam ser resolvidas:\n" .. table.concat(failures, "\n"))
  end
end
