-- Headless Vulkan test fixture
-- Creates a headless Vulkan context (no surface, no window) for offscreen rendering tests.

local vk = require("vkapi")
local ffi = require("ffi")
local bit = require("bit")

---@class HeadlessContext
---@field instance vk.Instance
---@field physicalDevice vk.ffi.PhysicalDevice
---@field device vk.Device
---@field queue vk.ffi.Queue
---@field queueFamilyIdx number
---@field commandPool vk.ffi.CommandPool
---@field memoryProperties vk.ffi.PhysicalDeviceMemoryProperties
---@field findMemoryType fun(self:HeadlessContext, typeBits: number, propertyFlags: number): number?

local M = {}

---Create a headless Vulkan context with instance, device, queue and command pool.
---@return HeadlessContext
function M.createContext()
	local instance = vk.createInstance({
		enabledExtensionNames = {},
		enabledLayerNames = {},
		applicationInfo = {
			name = "headless test",
			version = 1.0,
			engineName = "vkapi",
			engineVersion = 1.0,
			apiVersion = vk.ApiVersion.V1_0
		}
	})

	local devices = instance:enumeratePhysicalDevices()
	if #devices == 0 then
		error("No Vulkan physical devices found")
	end

	local physicalDevice = devices[1]
	local queueFamilyIdx = nil ---@type number?

	for idx, family in ipairs(vk.getPhysicalDeviceQueueFamilyProperties(physicalDevice)) do
		if bit.band(family.queueFlags, vk.QueueFlagBits.GRAPHICS) ~= 0 then
			queueFamilyIdx = idx - 1
			break
		end
	end

	assert(queueFamilyIdx, "No graphics queue family found")

	local device = instance:createDevice(physicalDevice, {
		enabledExtensionNames = {},
		queueCreateInfos = {
			{
				queueFamilyIndex = queueFamilyIdx,
				queuePriorities = { 1.0 },
				queueCount = 1
			}
		}
	})

	local queue = device:getDeviceQueue(queueFamilyIdx, 0)

	local commandPool = device:createCommandPool({
		queueFamilyIndex = queueFamilyIdx,
		flags = vk.CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER
	})

	local memoryProperties = vk.getPhysicalDeviceMemoryProperties(physicalDevice)

	local ctx = {
		instance = instance,
		physicalDevice = physicalDevice,
		device = device,
		queue = queue,
		queueFamilyIdx = queueFamilyIdx,
		commandPool = commandPool,
		memoryProperties = memoryProperties
	}

	---Find a memory type matching the given type bits and property flags.
	---@param typeBits number
	---@param propertyFlags number
	---@return number?
	function ctx:findMemoryType(typeBits, propertyFlags)
		for i = 0, self.memoryProperties.memoryTypeCount - 1 do
			local memType = self.memoryProperties.memoryTypes[i]
			if bit.band(typeBits, bit.lshift(1, i)) ~= 0
				and bit.band(memType.propertyFlags, propertyFlags) == propertyFlags then
				return i
			end
		end
		return nil
	end

	return ctx
end

---Allocate a staging buffer (HOST_VISIBLE | HOST_COHERENT) for readback.
---@param ctx HeadlessContext
---@param size number
---@return vk.ffi.Buffer, vk.ffi.DeviceMemory
function M.createStagingBuffer(ctx, size)
	local device = ctx.device
	local buffer = device:createBuffer({
		size = size,
		usage = vk.BufferUsageFlagBits.TRANSFER_DST,
		sharingMode = vk.SharingMode.EXCLUSIVE
	})

	local memRequirements = device:getBufferMemoryRequirements(buffer)
	local memTypeIndex = ctx:findMemoryType(memRequirements.memoryTypeBits,
		bit.bor(vk.MemoryPropertyFlagBits.HOST_VISIBLE, vk.MemoryPropertyFlagBits.HOST_COHERENT))
	assert(memTypeIndex, "No suitable host-visible memory type found for staging buffer")

	local memory = device:allocateMemory({
		allocationSize = memRequirements.size,
		memoryTypeIndex = memTypeIndex
	})
	device:bindBufferMemory(buffer, memory, 0)

	return buffer, memory
end

---Create a 2D color attachment image + view + render pass + framebuffer.
---@param ctx HeadlessContext
---@param width number
---@param height number
---@param format vk.Format
---@return vk.ffi.Image, vk.ffi.DeviceMemory, vk.ffi.ImageView, vk.ffi.RenderPass, vk.ffi.Framebuffer
function M.createColorAttachment(ctx, width, height, format)
	local device = ctx.device

	local image = device:createImage({
		imageType = vk.ImageType.TYPE_2D,
		format = format,
		extent = { width = width, height = height, depth = 1 },
		mipLevels = 1,
		arrayLayers = 1,
		samples = vk.SampleCountFlagBits.COUNT_1,
		tiling = vk.ImageTiling.OPTIMAL,
		usage = bit.bor(vk.ImageUsageFlagBits.COLOR_ATTACHMENT, vk.ImageUsageFlagBits.TRANSFER_SRC),
		sharingMode = vk.SharingMode.EXCLUSIVE,
		initialLayout = vk.ImageLayout.UNDEFINED
	})

	local memRequirements = device:getImageMemoryRequirements(image)
	local memTypeIndex = ctx:findMemoryType(memRequirements.memoryTypeBits, vk.MemoryPropertyFlagBits.DEVICE_LOCAL)
	assert(memTypeIndex, "No suitable device-local memory type found for image")

	local memory = device:allocateMemory({
		allocationSize = memRequirements.size,
		memoryTypeIndex = memTypeIndex
	})
	device:bindImageMemory(image, memory, 0)

	local imageView = device:createImageView({
		image = image,
		viewType = vk.ImageViewType.TYPE_2D,
		format = format,
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

	local renderPass = device:createRenderPass({
		attachments = {
			{
				format = format,
				samples = vk.SampleCountFlagBits.COUNT_1,
				loadOp = vk.AttachmentLoadOp.CLEAR,
				storeOp = vk.AttachmentStoreOp.STORE,
				stencilLoadOp = vk.AttachmentLoadOp.DONT_CARE,
				initialLayout = vk.ImageLayout.UNDEFINED,
				finalLayout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL
			}
		},
		subpasses = {
			{
				pipelineBindPoint = vk.PipelineBindPoint.GRAPHICS,
				colorAttachments = {
					{ attachment = 0, layout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL }
				}
			}
		}
	})

	local fbAttachments = vk.ImageViewArray(1)
	fbAttachments[0] = imageView
	local framebuffer = device:createFramebuffer({
		renderPass = renderPass,
		width = width,
		height = height,
		layers = 1,
		attachmentCount = 1,
		pAttachments = fbAttachments
	})

	return image, memory, imageView, renderPass, framebuffer
end

---Allocate a command buffer from the context's pool.
---@param ctx HeadlessContext
---@return vk.ffi.CommandBuffer
function M.allocateCommandBuffer(ctx)
	return ctx.device:allocateCommandBuffers({
		commandPool = ctx.commandPool,
		level = vk.CommandBufferLevel.PRIMARY,
		commandBufferCount = 1
	})[1]
end

---Submit a single command buffer and wait for it to complete.
---@param ctx HeadlessContext
---@param cmdBuffer vk.ffi.CommandBuffer
function M.submitAndWait(ctx, cmdBuffer)
	local device = ctx.device

	local fence = device:createFence({})

	local cmdBuffers = vk.CommandBufferArray(1)
	cmdBuffers[0] = cmdBuffer

	local submits = vk.SubmitInfoArray(1)
	submits[0] = vk.SubmitInfo({
		commandBufferCount = 1,
		pCommandBuffers = cmdBuffers
	})

	device:queueSubmit(ctx.queue, 1, submits, fence)

	local fences = vk.FenceArray(1)
	fences[0] = fence
	device:waitForFences(1, fences, true, math.huge)

	device:destroyFence(fence)
end

---Map staging buffer memory and return pixel data as a Lua table of RGBA bytes.
---@param device vk.Device
---@param memory vk.ffi.DeviceMemory
---@param size number
---@return table[] -- each entry: { r, g, b, a }
function M.readPixels(device, memory, size)
	local ptr = device:mapMemory(memory, 0, size)
	local pixels = {}
	local data = ffi.cast("uint8_t*", ptr)
	for i = 0, size - 1, 4 do
		pixels[#pixels + 1] = {
			r = data[i],
			g = data[i + 1],
			b = data[i + 2],
			a = data[i + 3]
		}
	end
	device:unmapMemory(memory)
	return pixels
end

return M
