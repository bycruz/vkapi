local vk = require("vkapi")
local winit = require("winit")
local ffi = require("ffi")

local desiredFormat = vk.Format.B8G8R8A8_SRGB
local pathSep = package.config:sub(1, 1)

local instance = vk.createInstance({
	enabledExtensionNames = { "VK_KHR_surface", ffi.os == "Linux" and "VK_KHR_xlib_surface" or "VK_KHR_win32_surface" },
	enabledLayerNames = {},
	applicationInfo = {
		name = "triangle test",
		version = 1.0,
		engineName = "vkapi",
		engineVersion = 1.0,
		apiVersion = vk.ApiVersion.V1_0
	}
})

local devices = instance:enumeratePhysicalDevices()
local physicalDevice = devices[1] -- Fallback to first

-- Prefer dgpu
for _, pd in ipairs(devices) do
	local props = vk.getPhysicalDeviceProperties(pd)
	if props.deviceType == vk.PhysicalDeviceType.DISCRETE_GPU then
		physicalDevice = pd
	end
end

---@type number?
local queueFamilyIdx

for idx, family in ipairs(vk.getPhysicalDeviceQueueFamilyProperties(physicalDevice)) do
	if bit.band(family.queueFlags, vk.QueueFlagBits.GRAPHICS) ~= 0 then
		queueFamilyIdx = idx - 1
		break
	end
end

if not queueFamilyIdx then
	error("No graphics queue family found")
end

local device = instance:createDevice(physicalDevice, {
	enabledExtensionNames = { "VK_KHR_swapchain" },
	queueCreateInfos = {
		{
			queueFamilyIndex = queueFamilyIdx,
			queuePriorities = { 1.0 },
			queueCount = 1
		}
	}
})

local eventLoop = winit.EventLoop.new()
local window = winit.Window.new(eventLoop, 800, 600)
eventLoop:register(window)

---@type vk.ffi.SurfaceKHR
local surface

if ffi.os == "Linux" then ---@cast window winit.x11.Window
	surface = instance:createXlibSurfaceKHR({
		dpy = window.display,
		window = window.id
	})
elseif ffi.os == "Windows" then ---@cast window winit.win32.Window
	surface = instance:createWin32SurfaceKHR({
		hinstance = window.id,
		hwnd = window.hwnd
	})
else
	error("Unsupported platform: " .. ffi.os)
end

local queue = device:getDeviceQueue(queueFamilyIdx, 0)

local renderPass = device:createRenderPass({
	attachments = {
		{
			format = desiredFormat,
			samples = vk.SampleCountFlagBits.COUNT_1,
			loadOp = vk.AttachmentLoadOp.CLEAR,
			storeOp = vk.AttachmentStoreOp.STORE,
			stencilLoadOp = vk.AttachmentLoadOp.DONT_CARE,
			initialLayout = vk.ImageLayout.UNDEFINED,
			finalLayout = vk.ImageLayout.PRESENT_SRC_KHR
		}
	},
	subpasses = {
		{
			pipelineBindPoint = vk.PipelineBindPoint.GRAPHICS,
			colorAttachments = {
				{
					layout = vk.ImageLayout.GENERAL,
					attachment = 0
				}
			}
		}
	}
})

local swapchain = device:createSwapchainKHR({
	surface = surface,
	minImageCount = 3,
	imageFormat = desiredFormat,
	imageColorSpace = vk.ColorSpaceKHR.SRGB_NONLINEAR,
	imageExtent = { width = window.width, height = window.height },
	imageArrayLayers = 1,
	imageUsage = vk.ImageUsageFlagBits.COLOR_ATTACHMENT,
	imageSharingMode = vk.SharingMode.EXCLUSIVE,
	preTransform = vk.SurfaceTransformFlagBitsKHR.IDENTITY,
	compositeAlpha = vk.CompositeAlphaFlagBitsKHR.OPAQUE,
	presentMode = vk.PresentModeKHR.IMMEDIATE,
	clipped = 1,
	oldSwapchain = nil
})

local swapchainImages = device:getSwapchainImagesKHR(swapchain)

local imageViews = {}
local framebuffers = {}

for i, image in ipairs(swapchainImages) do
	local imageView = device:createImageView({
		image = image,
		viewType = vk.ImageViewType.TYPE_2D,
		format = desiredFormat,
		subresourceRange = {
			aspectMask = vk.ImageAspectFlagBits.COLOR,
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1
		},
		components = {
			r = vk.ComponentSwizzle.IDENTITY,
			g = vk.ComponentSwizzle.IDENTITY,
			b = vk.ComponentSwizzle.IDENTITY,
			a = vk.ComponentSwizzle.IDENTITY
		}
	})

	imageViews[i] = imageView

	local attachments = vk.ImageViewArray(1)
	attachments[0] = imageView

	local framebuffer = device:createFramebuffer({
		renderPass = renderPass,
		width = window.width,
		height = window.height,
		layers = 1,
		attachmentCount = 1,
		pAttachments = attachments
	})

	framebuffers[i] = framebuffer
end

---@type vk.ffi.CommandPool[]
local commandPools = {}

---@type vk.ffi.CommandBuffer[]
local commandBuffers = {}

for i = 1, #swapchainImages do
	commandPools[i] = device:createCommandPool({
		queueFamilyIndex = queueFamilyIdx,
		flags = vk.CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER
	})

	commandBuffers[i] = device:allocateCommandBuffers({
		commandPool = commandPools[i],
		level = vk.CommandBufferLevel.PRIMARY,
		commandBufferCount = 1
	})[1]
end

local pipelineLayout = device:createPipelineLayout({})

local sourcePath = debug.getinfo(1, "S").source:sub(2):gsub("^@", "")
local targetFolder = sourcePath:match("(.*[/\\])") or ("." .. pathSep)
local projectFolder = targetFolder .. ".." .. pathSep .. ".." .. pathSep

local vertexModule ---@type vk.ffi.ShaderModule
do
	local code = io.open(projectFolder .. "shaders" .. pathSep .. "triangle.vert.spv", "rb"):read("*a")
	vertexModule = device:createShaderModule({
		codeSize = #code,
		pCode = ffi.cast("const uint32_t*", code)
	})
end

local fragmentModule ---@type vk.ffi.ShaderModule
do
	local code = io.open(projectFolder .. "shaders" .. pathSep .. "triangle.frag.spv", "rb"):read("*a")
	fragmentModule = device:createShaderModule({
		codeSize = #code,
		pCode = ffi.cast("const uint32_t*", code)
	})
end

local vertexBuffer = device:createBuffer({
	size = 3 * ffi.sizeof("float") * 7, -- 3 vertices, each with 3 floats for position and 4 for color
	usage = vk.BufferUsageFlagBits.VERTEX_BUFFER
})

do
	local memoryRequirements = device:getBufferMemoryRequirements(vertexBuffer)

	local memoryTypeIndex = nil ---@type number?
	local properties = vk.getPhysicalDeviceMemoryProperties(physicalDevice)
	for i = 0, properties.memoryTypeCount - 1 do
		local memType = properties.memoryTypes[i] ---@type vk.ffi.MemoryType
		if bit.band(memoryRequirements.memoryTypeBits, bit.lshift(1, i)) ~= 0 and
			bit.band(memType.propertyFlags, vk.MemoryPropertyFlagBits.HOST_VISIBLE) ~= 0 and
			bit.band(memType.propertyFlags, vk.MemoryPropertyFlagBits.HOST_COHERENT) ~= 0
		then
			memoryTypeIndex = i
			break
		end
	end

	local memoryChunk = device:allocateMemory({
		allocationSize = memoryRequirements.size,
		memoryTypeIndex = assert(memoryTypeIndex, "No suitable memory type found for vertex buffer")
	})

	device:bindBufferMemory(vertexBuffer, memoryChunk, 0)

	local data = ffi.cast("float*", device:mapMemory(memoryChunk, 0, memoryRequirements.size))
	-- vertex 0: top center, red
	data[0] = 0.0; data[1] = -0.5; data[2] = 0.0
	data[3] = 1.0; data[4] = 0.0; data[5] = 0.0; data[6] = 1.0
	-- vertex 1: bottom right, green
	data[7] = 0.5; data[8] = 0.5; data[9] = 0.0
	data[10] = 0.0; data[11] = 1.0; data[12] = 0.0; data[13] = 1.0
	-- vertex 2: bottom left, blue
	data[14] = -0.5; data[15] = 0.5; data[16] = 0.0
	data[17] = 0.0; data[18] = 0.0; data[19] = 1.0; data[20] = 1.0
	device:unmapMemory(memoryChunk)
end

local indexBuffer = device:createBuffer({
	size = 3 * ffi.sizeof("uint16_t"),
	usage = vk.BufferUsageFlagBits.INDEX_BUFFER
})


do
	local memoryRequirements = device:getBufferMemoryRequirements(indexBuffer)

	local memoryTypeIndex = nil ---@type number?
	local properties = vk.getPhysicalDeviceMemoryProperties(physicalDevice)
	for i = 0, properties.memoryTypeCount - 1 do
		local memType = properties.memoryTypes[i] ---@type vk.ffi.MemoryType
		if bit.band(memoryRequirements.memoryTypeBits, bit.lshift(1, i)) ~= 0 and
			bit.band(memType.propertyFlags, vk.MemoryPropertyFlagBits.HOST_VISIBLE) ~= 0 and
			bit.band(memType.propertyFlags, vk.MemoryPropertyFlagBits.HOST_COHERENT) ~= 0
		then
			memoryTypeIndex = i
			break
		end
	end

	local memoryChunk = device:allocateMemory({
		allocationSize = memoryRequirements.size,
		memoryTypeIndex = assert(memoryTypeIndex, "No suitable memory type found for index buffer")
	})

	device:bindBufferMemory(indexBuffer, memoryChunk, 0)

	local indices = ffi.cast("uint16_t*", device:mapMemory(memoryChunk, 0, memoryRequirements.size))
	indices[0] = 0; indices[1] = 1; indices[2] = 2
	device:unmapMemory(memoryChunk)
end

local pipeline = device:createGraphicsPipelines(nil, { {
	renderPass = renderPass,

	rasterizationState = {
		polygonMode = vk.PolygonMode.FILL,
		cullMode = vk.CullModeFlagBits.BACK,
		frontFace = vk.FrontFace.CLOCKWISE,
		lineWidth = 1.0
	},

	multisampleState = {
		rasterizationSamples = vk.SampleCountFlagBits.COUNT_1
	},

	vertexInputState = {
		attributes = {
			{
				location = 0,
				binding = 0,
				format = vk.Format.R32G32B32_SFLOAT,
				offset = 0
			},
			{
				location = 1,
				binding = 0,
				format = vk.Format.R32G32B32A32_SFLOAT,
				offset = 12
			}
		},
		bindings = {
			{
				binding = 0,
				stride = 28,
				inputRate = vk.VertexInputRate.VERTEX
			}
		}
	},

	inputAssemblyState = {
		topology = vk.PrimitiveTopology.TRIANGLE_LIST,
		primitiveRestartEnable = false
	},

	colorBlendState = {
		attachments = {
			{
				colorWriteMask = bit.bor(
					vk.ColorComponentFlagBits.R,
					vk.ColorComponentFlagBits.G,
					vk.ColorComponentFlagBits.B,
					vk.ColorComponentFlagBits.A
				),
				blendEnable = false
			}
		}
	},

	viewportState = {
		viewportCount = 1,
		scissorCount = 1
	},

	dynamicState = {
		dynamicStates = {
			vk.DynamicState.VIEWPORT,
			vk.DynamicState.SCISSOR
		}
	},

	stages = {
		{
			stage = vk.ShaderStageFlagBits.VERTEX,
			module = vertexModule
		},
		{
			stage = vk.ShaderStageFlagBits.FRAGMENT,
			module = fragmentModule
		}
	},
	layout = pipelineLayout
} })[1]

local scissors = vk.Rect2DArray(1)
scissors[0] = {
	offset = { x = 0, y = 0 },
	extent = { width = 800, height = 600 }
}

local viewports = vk.ViewportArray(1)
viewports[0] = {
	x = 0,
	y = 0,
	width = 800,
	height = 600,
	minDepth = 0.0,
	maxDepth = 1.0
}

local renderPassBeginInfo = vk.RenderPassBeginInfo()
renderPassBeginInfo.renderPass = renderPass
renderPassBeginInfo.framebuffer = framebuffers[1]
renderPassBeginInfo.renderArea = {
	offset = { x = 0, y = 0 },
	extent = { width = 800, height = 600 }
}
renderPassBeginInfo.clearValueCount = 1

local clearValueArray = vk.ClearValueArray(1) -- pin: ensure memory isn't freed
renderPassBeginInfo.pClearValues = clearValueArray
renderPassBeginInfo.pClearValues[0].color.float32[0] = 0.0
renderPassBeginInfo.pClearValues[0].color.float32[1] = 0.0
renderPassBeginInfo.pClearValues[0].color.float32[2] = 0.0
renderPassBeginInfo.pClearValues[0].color.float32[3] = 1.0

local imageAvailableSemaphores = {}
local renderFinishedSemaphores = {}
local inFlightFences = {}
for i = 1, #swapchainImages do
	imageAvailableSemaphores[i] = device:createSemaphore({})
	renderFinishedSemaphores[i] = device:createSemaphore({})
	inFlightFences[i] = device:createFence({ flags = vk.FenceCreateFlagBits.SIGNALED })
end

local swapchains = vk.SwapchainKHRArray(1)
swapchains[0] = swapchain

local imageIndices = ffi.new("uint32_t[1]")

local waitSemaphoresLen = #swapchainImages
local waitSemaphores = vk.SemaphoreArray(waitSemaphoresLen)
for i = 1, waitSemaphoresLen do
	waitSemaphores[i - 1] = imageAvailableSemaphores[i]
end

local signalSemaphoresLen = #swapchainImages
local signalSemaphores = vk.SemaphoreArray(signalSemaphoresLen)
for i = 1, signalSemaphoresLen do
	signalSemaphores[i - 1] = renderFinishedSemaphores[i]
end

local commandBuffersToSubmit = vk.CommandBufferArray(1)
local waitDstStageMask = ffi.new("uint32_t[1]", vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT) -- pin: ensure memory isn't freed

local queueSubmits = vk.SubmitInfoArray(1)
queueSubmits[0] = vk.SubmitInfo({
	waitSemaphoreCount = waitSemaphoresLen,
	pWaitSemaphores = waitSemaphores,
	pWaitDstStageMask = waitDstStageMask,
	commandBufferCount = 1,
	pCommandBuffers = commandBuffersToSubmit,
	signalSemaphoreCount = signalSemaphoresLen,
	pSignalSemaphores = signalSemaphores
})

local vertexBuffers = vk.BufferArray(1)
vertexBuffers[0] = vertexBuffer

local vertexBufferOffsets = vk.DeviceSizeArray(1)
vertexBufferOffsets[0] = 0

local fencesLen = #swapchainImages
local fences = vk.FenceArray(fencesLen)
for i = 1, fencesLen do
	fences[i - 1] = inFlightFences[i]
end

-- Track which semaphore was used to acquire each swapchain image
local imageAcquireSemaphoreForImage = {}
local waitSemaphoreForSubmit = vk.SemaphoreArray(1)
local beginInfo = vk.CommandBufferBeginInfo({
	flags = vk.CommandBufferUsageFlagBits.SIMULTANEOUS_USE
})

local function recreateSwapchain()
	device:queueWaitIdle(queue)

	for _, fb in ipairs(framebuffers) do device:destroyFramebuffer(fb) end
	for _, iv in ipairs(imageViews) do device:destroyImageView(iv) end

	local oldSwapchain = swapchain
	swapchain = device:createSwapchainKHR({
		surface = surface,
		minImageCount = 3,
		imageFormat = desiredFormat,
		imageColorSpace = vk.ColorSpaceKHR.SRGB_NONLINEAR,
		imageExtent = { width = window.width, height = window.height },
		imageArrayLayers = 1,
		imageUsage = vk.ImageUsageFlagBits.COLOR_ATTACHMENT,
		imageSharingMode = vk.SharingMode.EXCLUSIVE,
		preTransform = vk.SurfaceTransformFlagBitsKHR.IDENTITY,
		compositeAlpha = vk.CompositeAlphaFlagBitsKHR.OPAQUE,
		presentMode = vk.PresentModeKHR.IMMEDIATE,
		clipped = 1,
		oldSwapchain = oldSwapchain
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
				layerCount = 1
			},
			components = {
				r = vk.ComponentSwizzle.IDENTITY,
				g = vk.ComponentSwizzle.IDENTITY,
				b = vk.ComponentSwizzle.IDENTITY,
				a = vk.ComponentSwizzle.IDENTITY
			}
		})
		imageViews[i] = iv
		local attachments = vk.ImageViewArray(1)
		attachments[0] = iv
		framebuffers[i] = device:createFramebuffer({
			renderPass = renderPass,
			width = window.width,
			height = window.height,
			layers = 1,
			attachmentCount = 1,
			pAttachments = attachments
		})
	end

	scissors[0].extent.width = window.width
	scissors[0].extent.height = window.height
	viewports[0].width = window.width
	viewports[0].height = window.height
	renderPassBeginInfo.renderArea.extent.width = window.width
	renderPassBeginInfo.renderArea.extent.height = window.height
end

local currentFrame = 1
local function draw()
	local imgSemaphore = imageAvailableSemaphores[currentFrame]
	local fence = inFlightFences[currentFrame]

	local commandBuffer = commandBuffers[currentFrame]

	local frameOffset = currentFrame - 1
	currentFrame = currentFrame % fencesLen + 1

	device:waitForFences(1, fences + frameOffset, true, math.huge)

	-- Acquire the next image using the current frame's semaphore
	local acquireResult, imageIndex = device:acquireNextImageKHR(swapchain, -1, imgSemaphore, nil)
	if acquireResult == vk.Result.ERROR_OUT_OF_DATE_KHR then
		return
	end

	device:resetFences(1, fences + frameOffset)
	device:resetCommandBuffer(commandBuffer)

	commandBuffersToSubmit[0] = commandBuffer

	-- Track which semaphore was used for this image
	imageAcquireSemaphoreForImage[imageIndex + 1] = imgSemaphore

	local renderSemaphore = renderFinishedSemaphores[imageIndex + 1]

	-- Wait on the semaphore that was used to acquire this specific image
	waitSemaphoreForSubmit[0] = imageAcquireSemaphoreForImage[imageIndex + 1]
	queueSubmits[0].waitSemaphoreCount = 1
	queueSubmits[0].pWaitSemaphores = waitSemaphoreForSubmit

	queueSubmits[0].signalSemaphoreCount = 1
	queueSubmits[0].pSignalSemaphores = signalSemaphores + imageIndex

	renderPassBeginInfo.framebuffer = framebuffers[imageIndex + 1]
	imageIndices[0] = imageIndex

	device:beginCommandBuffer(commandBuffer, beginInfo)
	device:cmdSetScissor(commandBuffer, 0, 1, scissors)
	device:cmdSetViewport(commandBuffer, 0, 1, viewports)
	device:cmdBeginRenderPass(commandBuffer, renderPassBeginInfo, vk.SubpassContents.INLINE)
	device:cmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexBufferOffsets)
	device:cmdBindIndexBuffer(commandBuffer, indexBuffer, 0, vk.IndexType.UINT16)
	device:cmdBindPipeline(commandBuffer, vk.PipelineBindPoint.GRAPHICS, pipeline)
	device:cmdDraw(commandBuffer, 3, 1, 0, 0)
	device:cmdEndRenderPass(commandBuffer)
	device:endCommandBuffer(commandBuffer)

	device:queueSubmit(queue, 1, queueSubmits, fence)
	device:queuePresentKHR(queue, swapchain, imageIndex, renderSemaphore)
end

eventLoop:run(function(event, handler)
	if event.name == "redraw" then
		draw()
	elseif event.name == "resize" then
		recreateSwapchain()
	elseif event.name == "windowClose" then
		handler:exit()
	elseif event.name == "aboutToWait" then
		handler:requestRedraw(window)
	end
end)
