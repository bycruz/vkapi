local vk = require("vkapi")
local winit = require("winit")
local ffi = require("ffi")

local W, H = 1200, 720
local desiredFormat = vk.Format.B8G8R8A8_SRGB
local depthFormat = vk.Format.D32_SFLOAT
local pathSep = package.config:sub(1, 1)

-- ─── Vulkan bootstrap ────────────────────────────────────────────────────────

local instance = vk.createInstance({
	enabledExtensionNames = {
		"VK_KHR_surface",
		ffi.os == "Linux" and "VK_KHR_xlib_surface" or "VK_KHR_win32_surface",
	},
	enabledLayerNames = {},
	applicationInfo = {
		name = "voxel",
		version = 1.0,
		engineName = "vkapi",
		engineVersion = 1.0,
		apiVersion = vk.ApiVersion.V1_0,
	},
})

local devices = instance:enumeratePhysicalDevices()
local physicalDevice = devices[1]
for _, pd in ipairs(devices) do
	local props = vk.getPhysicalDeviceProperties(pd)
	if props.deviceType == vk.PhysicalDeviceType.DISCRETE_GPU then
		physicalDevice = pd
	end
end

local queueFamilyIdx
for idx, family in ipairs(vk.getPhysicalDeviceQueueFamilyProperties(physicalDevice)) do
	if bit.band(family.queueFlags, vk.QueueFlagBits.GRAPHICS) ~= 0 then
		queueFamilyIdx = idx - 1
		break
	end
end
assert(queueFamilyIdx, "No graphics queue family found")

local device = instance:createDevice(physicalDevice, {
	enabledExtensionNames = { "VK_KHR_swapchain" },
	queueCreateInfos = {
		{ queueFamilyIndex = queueFamilyIdx, queuePriorities = { 1.0 }, queueCount = 1 },
	},
})

local eventLoop = winit.EventLoop.new()
local window = winit.Window.new(eventLoop, W, H)
eventLoop:register(window)
window:setTitle("Voxel")

local surface
if ffi.os == "Linux" then
	---@cast window winit.x11.Window
	surface = instance:createXlibSurfaceKHR({ dpy = window.display, window = window.id })
elseif ffi.os == "Windows" then
	---@cast window winit.win32.Window
	surface = instance:createWin32SurfaceKHR({ hinstance = window.id, hwnd = window.hwnd })
else
	error("Unsupported platform: " .. ffi.os)
end

local queue = device:getDeviceQueue(queueFamilyIdx, 0)

-- ─── Helper: allocate device-local or host-visible memory ────────────────────

local function findMemoryType(requiredBits, propertyFlags)
	local props = vk.getPhysicalDeviceMemoryProperties(physicalDevice)
	for i = 0, props.memoryTypeCount - 1 do
		local mt = props.memoryTypes[i]
		if bit.band(requiredBits, bit.lshift(1, i)) ~= 0
			and bit.band(mt.propertyFlags, propertyFlags) == propertyFlags
		then
			return i
		end
	end
	error("No suitable memory type found")
end

local HOST_FLAGS   = bit.bor(vk.MemoryPropertyFlagBits.HOST_VISIBLE, vk.MemoryPropertyFlagBits.HOST_COHERENT)
local DEVICE_FLAGS = vk.MemoryPropertyFlagBits.DEVICE_LOCAL

local function createHostBuffer(size, usage)
	local buf = device:createBuffer({ size = size, usage = usage })
	local req = device:getBufferMemoryRequirements(buf)
	local mem = device:allocateMemory({
		allocationSize = req.size,
		memoryTypeIndex = findMemoryType(req.memoryTypeBits, HOST_FLAGS)
	})
	device:bindBufferMemory(buf, mem, 0)
	return buf, mem, req.size
end

local function createDeviceBuffer(size, usage)
	local buf = device:createBuffer({ size = size, usage = bit.bor(usage, vk.BufferUsageFlagBits.TRANSFER_DST) })
	local req = device:getBufferMemoryRequirements(buf)
	local mem = device:allocateMemory({
		allocationSize = req.size,
		memoryTypeIndex = findMemoryType(req.memoryTypeBits, DEVICE_FLAGS)
	})
	device:bindBufferMemory(buf, mem, 0)
	return buf
end

-- ─── Depth image ─────────────────────────────────────────────────────────────

---@param w number
---@param h number
---@return vk.ffi.Image
---@return vk.ffi.DeviceMemory
---@return vk.ffi.ImageView
local function createDepthResources(w, h)
	local img = device:createImage({
		imageType = vk.ImageType.TYPE_2D,
		format = depthFormat,
		extent = { width = w, height = h, depth = 1 },
		mipLevels = 1,
		arrayLayers = 1,
		samples = vk.SampleCountFlagBits.COUNT_1,
		tiling = vk.ImageTiling.OPTIMAL,
		usage = vk.ImageUsageFlagBits.DEPTH_STENCIL_ATTACHMENT,
		sharingMode = vk.SharingMode.EXCLUSIVE,
		initialLayout = vk.ImageLayout.UNDEFINED,
	})
	local req = device:getImageMemoryRequirements(img)
	local mem = device:allocateMemory({
		allocationSize = req.size,
		memoryTypeIndex = findMemoryType(req.memoryTypeBits, vk.MemoryPropertyFlagBits.DEVICE_LOCAL),
	})
	device:bindImageMemory(img, mem, 0)
	local view = device:createImageView({
		image = img,
		viewType = vk.ImageViewType.TYPE_2D,
		format = depthFormat,
		subresourceRange = {
			aspectMask = vk.ImageAspectFlagBits.DEPTH,
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		components = {
			r = vk.ComponentSwizzle.IDENTITY,
			g = vk.ComponentSwizzle.IDENTITY,
			b = vk.ComponentSwizzle.IDENTITY,
			a = vk.ComponentSwizzle.IDENTITY,
		},
	})
	return img, mem, view
end

local depthImage, depthMemory, depthImageView = createDepthResources(W, H)

-- ─── Render pass ─────────────────────────────────────────────────────────────

local renderPass = device:createRenderPass({
	attachments = {
		{
			format = desiredFormat,
			samples = vk.SampleCountFlagBits.COUNT_1,
			loadOp = vk.AttachmentLoadOp.CLEAR,
			storeOp = vk.AttachmentStoreOp.STORE,
			stencilLoadOp = vk.AttachmentLoadOp.DONT_CARE,
			initialLayout = vk.ImageLayout.UNDEFINED,
			finalLayout = vk.ImageLayout.PRESENT_SRC_KHR,
		},
		{
			format = depthFormat,
			samples = vk.SampleCountFlagBits.COUNT_1,
			loadOp = vk.AttachmentLoadOp.CLEAR,
			storeOp = vk.AttachmentStoreOp.DONT_CARE,
			stencilLoadOp = vk.AttachmentLoadOp.DONT_CARE,
			initialLayout = vk.ImageLayout.UNDEFINED,
			finalLayout = vk.ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		},
	},
	subpasses = {
		{
			pipelineBindPoint = vk.PipelineBindPoint.GRAPHICS,
			colorAttachments = {
				{ attachment = 0, layout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL },
			},
			depthStencilAttachment = {
				attachment = 1,
				layout = vk.ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
		},
	},
	dependencies = {
		{
			srcSubpass = 4294967295, -- VK_SUBPASS_EXTERNAL
			dstSubpass = 0,
			srcStageMask = bit.bor(vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT,
				vk.PipelineStageFlagBits.EARLY_FRAGMENT_TESTS),
			dstStageMask = bit.bor(vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT,
				vk.PipelineStageFlagBits.EARLY_FRAGMENT_TESTS),
			srcAccessMask = 0,
			dstAccessMask = bit.bor(vk.AccessFlags.COLOR_ATTACHMENT_WRITE, vk.AccessFlags.DEPTH_STENCIL_ATTACHMENT_WRITE),
		},
	},
})

-- ─── Swapchain ───────────────────────────────────────────────────────────────

local swapchain = device:createSwapchainKHR({
	surface = surface,
	minImageCount = 3,
	imageFormat = desiredFormat,
	imageColorSpace = vk.ColorSpaceKHR.SRGB_NONLINEAR,
	imageExtent = { width = W, height = H },
	imageArrayLayers = 1,
	imageUsage = vk.ImageUsageFlagBits.COLOR_ATTACHMENT,
	imageSharingMode = vk.SharingMode.EXCLUSIVE,
	preTransform = vk.SurfaceTransformFlagBitsKHR.IDENTITY,
	compositeAlpha = vk.CompositeAlphaFlagBitsKHR.OPAQUE,
	presentMode = vk.PresentModeKHR.IMMEDIATE,
	clipped = 1,
	oldSwapchain = nil,
})

local swapchainImages = device:getSwapchainImagesKHR(swapchain)
local imageViews = {}
local framebuffers = {}

for i, image in ipairs(swapchainImages) do
	local iv = device:createImageView({
		image = image,
		viewType = vk.ImageViewType.TYPE_2D,
		format = desiredFormat,
		subresourceRange = {
			aspectMask = vk.ImageAspectFlagBits.COLOR,
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		components = {
			r = vk.ComponentSwizzle.IDENTITY,
			g = vk.ComponentSwizzle.IDENTITY,
			b = vk.ComponentSwizzle.IDENTITY,
			a = vk.ComponentSwizzle.IDENTITY,
		},
	})
	imageViews[i] = iv

	local attachments = vk.ImageViewArray(2)
	attachments[0] = iv
	attachments[1] = depthImageView

	framebuffers[i] = device:createFramebuffer({
		renderPass = renderPass,
		width = W,
		height = H,
		layers = 1,
		attachmentCount = 2,
		pAttachments = attachments,
	})
end

-- ─── Command pools & buffers ─────────────────────────────────────────────────

local commandPools = {}
local commandBuffers = {}
for i = 1, #swapchainImages do
	commandPools[i] = device:createCommandPool({
		queueFamilyIndex = queueFamilyIdx,
		flags = vk.CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER,
	})
	commandBuffers[i] = device:allocateCommandBuffers({
		commandPool = commandPools[i],
		level = vk.CommandBufferLevel.PRIMARY,
		commandBufferCount = 1,
	})[1]
end

-- ─── Shaders ─────────────────────────────────────────────────────────────────

local sourcePath = debug.getinfo(1, "S").source:sub(2):gsub("^@", "")
local srcFolder = sourcePath:match("(.*[/\\])") or ("." .. pathSep)
local projectFolder = srcFolder .. ".." .. pathSep .. ".." .. pathSep

local function loadShader(path)
	local f = assert(io.open(path, "rb"), "Cannot open shader: " .. path)
	local code = f:read("*a")
	f:close()
	return device:createShaderModule({ codeSize = #code, pCode = ffi.cast("const uint32_t*", code) })
end

local vertModule = loadShader(projectFolder .. "shaders" .. pathSep .. "voxel.vert.spv")
local fragModule = loadShader(projectFolder .. "shaders" .. pathSep .. "voxel.frag.spv")

-- ─── Pipeline layout (push constants for MVP) ────────────────────────────────

local pushConstantRanges = ffi.new("VkPushConstantRange[1]")
pushConstantRanges[0].stageFlags = bit.bor(vk.ShaderStageFlagBits.VERTEX, vk.ShaderStageFlagBits.FRAGMENT)
pushConstantRanges[0].offset = 0
pushConstantRanges[0].size = 64 -- mat4

local pipelineLayout = device:createPipelineLayout({
	pushConstantRangeCount = 1,
	pPushConstantRanges = pushConstantRanges,
})

-- ─── Graphics pipeline ───────────────────────────────────────────────────────

-- stride = vec3 pos (12) + vec3 normal (12) + vec3 color (12) = 36
local pipeline = device:createGraphicsPipelines(nil, { {
	renderPass         = renderPass,
	layout             = pipelineLayout,

	stages             = {
		{ stage = vk.ShaderStageFlagBits.VERTEX,   module = vertModule },
		{ stage = vk.ShaderStageFlagBits.FRAGMENT, module = fragModule },
	},

	vertexInputState   = {
		bindings = {
			{ binding = 0, stride = 36, inputRate = vk.VertexInputRate.VERTEX },
		},
		attributes = {
			{ location = 0, binding = 0, format = vk.Format.R32G32B32_SFLOAT, offset = 0 },
			{ location = 1, binding = 0, format = vk.Format.R32G32B32_SFLOAT, offset = 12 },
			{ location = 2, binding = 0, format = vk.Format.R32G32B32_SFLOAT, offset = 24 },
		},
	},

	inputAssemblyState = {
		topology = vk.PrimitiveTopology.TRIANGLE_LIST,
		primitiveRestartEnable = false,
	},

	rasterizationState = {
		polygonMode = vk.PolygonMode.FILL,
		cullMode    = vk.CullModeFlagBits.NONE,
		frontFace   = vk.FrontFace.COUNTER_CLOCKWISE,
		lineWidth   = 1.0,
	},

	multisampleState   = {
		rasterizationSamples = vk.SampleCountFlagBits.COUNT_1,
	},

	depthStencilState  = {
		depthTestEnable  = true,
		depthWriteEnable = true,
		depthCompareOp   = vk.CompareOp.LESS,
	},

	colorBlendState    = {
		attachments = {
			{
				colorWriteMask = bit.bor(
					vk.ColorComponentFlagBits.R, vk.ColorComponentFlagBits.G,
					vk.ColorComponentFlagBits.B, vk.ColorComponentFlagBits.A
				),
				blendEnable = false,
			},
		},
	},

	viewportState      = { viewportCount = 1, scissorCount = 1 },
	dynamicState       = { dynamicStates = { vk.DynamicState.VIEWPORT, vk.DynamicState.SCISSOR } },
} })[1]

-- ─── World generation ─────────────────────────────────────────────────────────

local CX, CY, CZ = 256, 64, 256

local world = {}
for x = 0, CX - 1 do
	world[x] = {}
	for y = 0, CY - 1 do
		world[x][y] = {}
		for z = 0, CZ - 1 do
			local height = 12
				+ math.floor(8 * math.sin(x * 0.04) * math.cos(z * 0.04))
				+ math.floor(4 * math.sin(x * 0.015 + 1.2) * math.cos(z * 0.02 + 0.7))
				+ math.floor(2 * math.sin(x * 0.08 + 3.1) * math.cos(z * 0.07 + 2.0))
			world[x][y][z] = (y <= height) and 1 or 0
		end
	end
end

local function getVoxel(x, y, z)
	if x < 0 or x >= CX or y < 0 or y >= CY or z < 0 or z >= CZ then return 1 end
	return world[x][y][z]
end

-- Per-face colors: +X, -X, +Y, -Y, +Z, -Z
local faceColors   = {
	{ 0.60, 0.38, 0.20 }, -- +X
	{ 0.55, 0.34, 0.18 }, -- -X
	{ 0.30, 0.72, 0.22 }, -- +Y (top = grass)
	{ 0.50, 0.30, 0.15 }, -- -Y
	{ 0.58, 0.36, 0.19 }, -- +Z
	{ 0.53, 0.33, 0.17 }, -- -Z
}

-- face: vertices (4) as offsets, normal, color index
local faces        = {
	-- +X
	{ verts = { { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 }, { 1, 0, 1 } }, nx = 1,  ny = 0,  nz = 0,  ci = 1 },
	-- -X
	{ verts = { { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 0, 0, 0 } }, nx = -1, ny = 0,  nz = 0,  ci = 2 },
	-- +Y
	{ verts = { { 0, 1, 0 }, { 0, 1, 1 }, { 1, 1, 1 }, { 1, 1, 0 } }, nx = 0,  ny = 1,  nz = 0,  ci = 3 },
	-- -Y
	{ verts = { { 0, 0, 1 }, { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 } }, nx = 0,  ny = -1, nz = 0,  ci = 4 },
	-- +Z
	{ verts = { { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 1 } }, nx = 0,  ny = 0,  nz = 1,  ci = 5 },
	-- -Z
	{ verts = { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } }, nx = 0,  ny = 0,  nz = -1, ci = 6 },
}

local neighborDirs = {
	{ 1, 0, 0 }, { -1, 0, 0 },
	{ 0, 1, 0 }, { 0, -1, 0 },
	{ 0, 0, 1 }, { 0, 0, -1 },
}

local vertices     = {}
local indices      = {}
local vertCount    = 0

for x = 0, CX - 1 do
	for y = 0, CY - 1 do
		for z = 0, CZ - 1 do
			if world[x][y][z] == 1 then
				for fi, face in ipairs(faces) do
					local nd = neighborDirs[fi]
					if getVoxel(x + nd[1], y + nd[2], z + nd[3]) == 0 then
						local c = faceColors[face.ci]
						for _, v in ipairs(face.verts) do
							vertices[#vertices + 1] = x + v[1]
							vertices[#vertices + 1] = y + v[2]
							vertices[#vertices + 1] = z + v[3]
							vertices[#vertices + 1] = face.nx
							vertices[#vertices + 1] = face.ny
							vertices[#vertices + 1] = face.nz
							vertices[#vertices + 1] = c[1]
							vertices[#vertices + 1] = c[2]
							vertices[#vertices + 1] = c[3]
						end
						local base = vertCount
						indices[#indices + 1] = base + 0
						indices[#indices + 1] = base + 1
						indices[#indices + 1] = base + 2
						indices[#indices + 1] = base + 0
						indices[#indices + 1] = base + 2
						indices[#indices + 1] = base + 3
						vertCount = vertCount + 4
					end
				end
			end
		end
	end
end

local indexCount = #indices

-- ─── Upload mesh ─────────────────────────────────────────────────────────────

local vbSize = #vertices * ffi.sizeof("float")
local ibSize = #indices  * ffi.sizeof("uint32_t")

local stagingVB, stagingVBMem = createHostBuffer(vbSize, vk.BufferUsageFlagBits.TRANSFER_SRC)
local stagingIB, stagingIBMem = createHostBuffer(ibSize, vk.BufferUsageFlagBits.TRANSFER_SRC)

do
	local ptr = ffi.cast("float*",    device:mapMemory(stagingVBMem, 0, vbSize))
	for i, v in ipairs(vertices) do ptr[i - 1] = v end
	device:unmapMemory(stagingVBMem)
end
do
	local ptr = ffi.cast("uint32_t*", device:mapMemory(stagingIBMem, 0, ibSize))
	for i, v in ipairs(indices)  do ptr[i - 1] = v end
	device:unmapMemory(stagingIBMem)
end

local vertexBuffer = createDeviceBuffer(vbSize, vk.BufferUsageFlagBits.VERTEX_BUFFER)
local indexBuffer  = createDeviceBuffer(ibSize, vk.BufferUsageFlagBits.INDEX_BUFFER)

do
	local transferPool = device:createCommandPool({ queueFamilyIndex = queueFamilyIdx })
	local cb = device:allocateCommandBuffers({
		commandPool = transferPool,
		level = vk.CommandBufferLevel.PRIMARY,
		commandBufferCount = 1,
	})[1]

	device:beginCommandBuffer(cb, vk.CommandBufferBeginInfo({
		flags = vk.CommandBufferUsageFlagBits.ONE_TIME_SUBMIT
	}))

	local vbRegion = vk.BufferCopyArray(1)
	vbRegion[0] = { srcOffset = 0, dstOffset = 0, size = vbSize }
	local ibRegion = vk.BufferCopyArray(1)
	ibRegion[0] = { srcOffset = 0, dstOffset = 0, size = ibSize }
	device:cmdCopyBuffer(cb, stagingVB, vertexBuffer, 1, vbRegion)
	device:cmdCopyBuffer(cb, stagingIB, indexBuffer,  1, ibRegion)

	device:endCommandBuffer(cb)

	local submitCBs = vk.CommandBufferArray(1)
	submitCBs[0] = cb
	local transferSubmit = vk.SubmitInfoArray(1)
	transferSubmit[0] = vk.SubmitInfo({ commandBufferCount = 1, pCommandBuffers = submitCBs })
	device:queueSubmit(queue, 1, transferSubmit, nil)
	device:queueWaitIdle(queue)
end

-- ─── Sync objects ─────────────────────────────────────────────────────────────

local imageAvailableSemaphores = {}
local renderFinishedSemaphores = {}
local inFlightFences = {}
for i = 1, #swapchainImages do
	imageAvailableSemaphores[i] = device:createSemaphore({})
	renderFinishedSemaphores[i] = device:createSemaphore({})
	inFlightFences[i] = device:createFence({ flags = vk.FenceCreateFlagBits.SIGNALED })
end

-- ─── Math helpers ─────────────────────────────────────────────────────────────

-- Matrices are stored column-major (index [col*4+row+1]) to match GLSL std140.

local function mat4Mul(a, b)
	local r = {}
	for col = 0, 3 do
		for row = 0, 3 do
			local s = 0
			for k = 0, 3 do
				s = s + a[k * 4 + row + 1] * b[col * 4 + k + 1]
			end
			r[col * 4 + row + 1] = s
		end
	end
	return r
end

local function mat4Perspective(fovY, aspect, near, far)
	local f = 1.0 / math.tan(fovY * 0.5)
	-- column-major: col0, col1, col2, col3
	return {
		f / aspect, 0, 0, 0,            -- col 0
		0, f, 0, 0,                     -- col 1
		0, 0, far / (near - far), -1,   -- col 2
		0, 0, (near * far) / (near - far), 0, -- col 3
	}
end

local function mat4LookAt(ex, ey, ez, cx, cy, cz, ux, uy, uz)
	local fx, fy, fz = cx - ex, cy - ey, cz - ez
	local len = math.sqrt(fx * fx + fy * fy + fz * fz)
	fx, fy, fz = fx / len, fy / len, fz / len

	local rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
	len = math.sqrt(rx * rx + ry * ry + rz * rz)
	rx, ry, rz = rx / len, ry / len, rz / len

	local ux2, uy2, uz2 = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx

	-- column-major view matrix
	return {
		rx, ux2, -fx, 0, -- col 0
		ry, uy2, -fy, 0, -- col 1
		rz, uz2, -fz, 0, -- col 2
		-(rx * ex + ry * ey + rz * ez),
		-(ux2 * ex + uy2 * ey + uz2 * ez),
		(fx * ex + fy * ey + fz * ez),
		1, -- col 3
	}
end

-- ─── Camera state ─────────────────────────────────────────────────────────────

local cam = {
	x = CX * 0.5,
	y = 30,
	z = CZ * 0.5 + 50,
	yaw = math.pi, -- looking toward -Z initially
	pitch = -0.25,
	speed = 10.0,
}

local keys = {}
local cursorGrabbed = false
local lastTime = os.clock()
local dt = 0

local function camForward()
	return math.sin(cam.yaw) * math.cos(cam.pitch),
		math.sin(cam.pitch),
		-math.cos(cam.yaw) * math.cos(cam.pitch)
end

local function camRight()
	return math.cos(cam.yaw), 0, math.sin(cam.yaw)
end

local function updateCamera(dt)
	local fx, fy, fz = camForward()
	local rx, _, rz = camRight()

	local dx, dy, dz = 0, 0, 0

	if keys["w"] then
		dx = dx + fx; dy = dy + fy; dz = dz + fz
	end
	if keys["s"] then
		dx = dx - fx; dy = dy - fy; dz = dz - fz
	end
	if keys["a"] then
		dx = dx - rx; dz = dz - rz
	end
	if keys["d"] then
		dx = dx + rx; dz = dz + rz
	end
	if keys[" "] then dy = dy - 1 end

	local len = math.sqrt(dx * dx + dy * dy + dz * dz)
	if len > 0 then
		dx, dy, dz = dx / len, dy / len, dz / len
	end

	local speedMul = keys["left-shift"] and 3.0 or keys["left-ctrl"] and 0.33 or 1.0
	cam.x = cam.x + dx * cam.speed * speedMul * dt
	cam.y = cam.y + dy * cam.speed * speedMul * dt
	cam.z = cam.z + dz * cam.speed * speedMul * dt
end

local function buildMVP()
	local fx, fy, fz = camForward()
	local view = mat4LookAt(
		cam.x, cam.y, cam.z,
		cam.x + fx, cam.y + fy, cam.z + fz,
		0, 1, 0
	)
	local proj = mat4Perspective(math.rad(70), W / H, 0.1, 500.0)
	return mat4Mul(proj, view)
end

-- ─── Draw state ───────────────────────────────────────────────────────────────

local scissors = vk.Rect2DArray(1)
scissors[0] = { offset = { x = 0, y = 0 }, extent = { width = W, height = H } }

local viewports = vk.ViewportArray(1)
viewports[0] = { x = 0, y = 0, width = W, height = H, minDepth = 0.0, maxDepth = 1.0 }

local clearValues = vk.ClearValueArray(2)
clearValues[0].color.float32[0] = 0.53
clearValues[0].color.float32[1] = 0.81
clearValues[0].color.float32[2] = 0.98
clearValues[0].color.float32[3] = 1.0
clearValues[1].depthStencil.depth = 1.0
clearValues[1].depthStencil.stencil = 0

local rpBeginInfo = vk.RenderPassBeginInfo()
rpBeginInfo.renderPass = renderPass
rpBeginInfo.renderArea = { offset = { x = 0, y = 0 }, extent = { width = W, height = H } }
rpBeginInfo.clearValueCount = 2
rpBeginInfo.pClearValues = clearValues

local mvpData = ffi.new("float[16]")

local fencesLen = #swapchainImages
local fences = vk.FenceArray(fencesLen)
for i = 1, fencesLen do fences[i - 1] = inFlightFences[i] end

local signalSemaphores = vk.SemaphoreArray(fencesLen)
for i = 1, fencesLen do signalSemaphores[i - 1] = renderFinishedSemaphores[i] end

local imageAcquireSemaphoreForImage = {}
local waitSemaphoreForSubmit = vk.SemaphoreArray(1)
local commandBuffersToSubmit = vk.CommandBufferArray(1)
local imageIndices = ffi.new("uint32_t[1]")
local waitDstStageMask = ffi.new("uint32_t[1]", vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT)

local queueSubmits = vk.SubmitInfoArray(1)
queueSubmits[0] = vk.SubmitInfo({
	waitSemaphoreCount = 1,
	pWaitSemaphores = waitSemaphoreForSubmit,
	pWaitDstStageMask = waitDstStageMask,
	commandBufferCount = 1,
	pCommandBuffers = commandBuffersToSubmit,
	signalSemaphoreCount = 1,
	pSignalSemaphores = signalSemaphores,
})

local vertexBuffers = vk.BufferArray(1)
vertexBuffers[0] = vertexBuffer
local vertexOffsets = vk.DeviceSizeArray(1)
vertexOffsets[0] = 0

local beginInfo = vk.CommandBufferBeginInfo({ flags = vk.CommandBufferUsageFlagBits.SIMULTANEOUS_USE })

local function recreateSwapchain()
	device:queueWaitIdle(queue)

	W = window.width
	H = window.height

	for _, fb in ipairs(framebuffers) do device:destroyFramebuffer(fb) end
	for _, iv in ipairs(imageViews) do device:destroyImageView(iv) end
	device:destroyImageView(depthImageView)
	device:destroyImage(depthImage)
	device:freeMemory(depthMemory)

	depthImage, depthMemory, depthImageView = createDepthResources(W, H)

	local oldSwapchain = swapchain
	swapchain = device:createSwapchainKHR({
		surface = surface,
		minImageCount = 3,
		imageFormat = desiredFormat,
		imageColorSpace = vk.ColorSpaceKHR.SRGB_NONLINEAR,
		imageExtent = { width = W, height = H },
		imageArrayLayers = 1,
		imageUsage = vk.ImageUsageFlagBits.COLOR_ATTACHMENT,
		imageSharingMode = vk.SharingMode.EXCLUSIVE,
		preTransform = vk.SurfaceTransformFlagBitsKHR.IDENTITY,
		compositeAlpha = vk.CompositeAlphaFlagBitsKHR.OPAQUE,
		presentMode = vk.PresentModeKHR.IMMEDIATE,
		clipped = 1,
		oldSwapchain = oldSwapchain,
	})
	device:destroySwapchainKHR(oldSwapchain)

	imageViews = {}
	framebuffers = {}
	for i, image in ipairs(device:getSwapchainImagesKHR(swapchain)) do
		local iv = device:createImageView({
			image = image,
			viewType = vk.ImageViewType.TYPE_2D,
			format = desiredFormat,
			subresourceRange = {
				aspectMask = vk.ImageAspectFlagBits.COLOR,
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
			components = {
				r = vk.ComponentSwizzle.IDENTITY,
				g = vk.ComponentSwizzle.IDENTITY,
				b = vk.ComponentSwizzle.IDENTITY,
				a = vk.ComponentSwizzle.IDENTITY,
			},
		})
		imageViews[i] = iv
		local attachments = vk.ImageViewArray(2)
		attachments[0] = iv
		attachments[1] = depthImageView
		framebuffers[i] = device:createFramebuffer({
			renderPass = renderPass,
			width = W,
			height = H,
			layers = 1,
			attachmentCount = 2,
			pAttachments = attachments,
		})
	end

	scissors[0].extent.width = W
	scissors[0].extent.height = H
	viewports[0].width = W
	viewports[0].height = H
	rpBeginInfo.renderArea.extent.width = W
	rpBeginInfo.renderArea.extent.height = H
end

local currentFrame = 1

local function draw()
	local now = os.clock()
	dt = now - lastTime
	lastTime = now

	updateCamera(dt)

	local mvp = buildMVP()
	for i = 0, 15 do mvpData[i] = mvp[i + 1] end

	local fence = inFlightFences[currentFrame]
	local imgSemaphore = imageAvailableSemaphores[currentFrame]
	local cb = commandBuffers[currentFrame]
	local frameOffset = currentFrame - 1
	currentFrame = currentFrame % fencesLen + 1

	device:waitForFences(1, fences + frameOffset, true, math.huge)

	local acquireResult, imageIndex = device:acquireNextImageKHR(swapchain, -1, imgSemaphore, nil)
	if acquireResult == vk.Result.ERROR_OUT_OF_DATE_KHR then
		recreateSwapchain()
		return
	end

	device:resetFences(1, fences + frameOffset)
	device:resetCommandBuffer(cb)

	commandBuffersToSubmit[0] = cb
	imageAcquireSemaphoreForImage[imageIndex + 1] = imgSemaphore

	waitSemaphoreForSubmit[0] = imageAcquireSemaphoreForImage[imageIndex + 1]
	queueSubmits[0].pWaitSemaphores = waitSemaphoreForSubmit
	queueSubmits[0].pSignalSemaphores = signalSemaphores + imageIndex

	rpBeginInfo.framebuffer = framebuffers[imageIndex + 1]
	imageIndices[0] = imageIndex

	device:beginCommandBuffer(cb, beginInfo)
	device:cmdSetScissor(cb, 0, 1, scissors)
	device:cmdSetViewport(cb, 0, 1, viewports)
	device:cmdBeginRenderPass(cb, rpBeginInfo, vk.SubpassContents.INLINE)
	device:cmdBindPipeline(cb, vk.PipelineBindPoint.GRAPHICS, pipeline)
	device:cmdPushConstants(cb, pipelineLayout, vk.ShaderStageFlagBits.VERTEX, 0, 64, mvpData)
	device:cmdBindVertexBuffers(cb, 0, 1, vertexBuffers, vertexOffsets)
	device:cmdBindIndexBuffer(cb, indexBuffer, 0, vk.IndexType.UINT32)
	device:cmdDrawIndexed(cb, indexCount, 1, 0, 0, 0)
	device:cmdEndRenderPass(cb)
	device:endCommandBuffer(cb)

	device:queueSubmit(queue, 1, queueSubmits, fence)
	local presentResult = device:queuePresentKHR(queue, swapchain, imageIndex, renderFinishedSemaphores[imageIndex + 1])
	if presentResult == vk.Result.ERROR_OUT_OF_DATE_KHR or presentResult == vk.Result.SUBOPTIMAL_KHR then
		recreateSwapchain()
	end
end

-- ─── Event loop ───────────────────────────────────────────────────────────────

eventLoop:run(function(event, handler)
	handler:setMode("poll")

	if event.name == "redraw" then
		draw()
	elseif event.name == "resize" then
		recreateSwapchain()
	elseif event.name == "windowClose" then
		handler:exit()
	elseif event.name == "aboutToWait" then
		handler:requestRedraw(window)
	elseif event.name == "keyPress" then
		keys[event.key] = true
		if event.key == "escape" then
			if cursorGrabbed then
				cursorGrabbed = false
				window:setCursorGrab("none")
			else
				handler:exit()
			end
		end
	elseif event.name == "keyRelease" then
		keys[event.key] = false
	elseif event.name == "mousePress" then
		if event.button == 1 and not cursorGrabbed then
			cursorGrabbed = true
			window:setCursorGrab("locked")
		end
	elseif event.name == "mouseMotion" then
		if cursorGrabbed then
			cam.yaw   = cam.yaw + event.dx * 0.003
			cam.pitch = cam.pitch + event.dy * 0.003
			cam.pitch = math.max(-math.pi * 0.49, math.min(math.pi * 0.49, cam.pitch))
		end
	elseif event.name == "focusOut" then
		if cursorGrabbed then
			cursorGrabbed = false
			window:setCursorGrab("none")
		end
	end
end)
