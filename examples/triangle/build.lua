local build = require("lde-build")

local sep = string.sub(package.config, 1, 1)

--- Build scripts are loaded from a string chunk, so
--- debug.getinfo(1, "S").source is the script's text, not its path.
--- build.outDir is the absolute path of target/<package>; the package root
--- sits two levels above it. src/init.lua resolves its shaders relative to the
--- package root, so the compiled SPIR-V has to be written back there.
local packageRoot = build.outDir .. sep .. ".." .. sep .. ".."
local shadersDir = packageRoot .. sep .. "shaders"

--- lde only hashes src/, lde.json and build.lua to decide whether to re-run
--- this script, so a "only compile if the .spv is missing" guard would ship
--- stale SPIR-V after a shader edit. Shaders are tiny; always rebuild them.
---@param stage "vert" | "frag"
---@param name string
local function glslToSpirv(stage, name)
	local input = shadersDir .. sep .. name .. "." .. stage .. ".glsl"
	local output = shadersDir .. sep .. name .. "." .. stage .. ".spv"

	local command = string.format("glslc -fshader-stage=%s \"%s\" -o \"%s\"", stage, input, output)
	local ok, err = pcall(function()
		build:sh(command)
	end)
	if not ok then
		error("Failed to compile GLSL shader: " .. input .. "\n" .. tostring(err))
	end
end

glslToSpirv("vert", "triangle")
glslToSpirv("frag", "triangle")
