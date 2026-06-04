local test = require("lde-test")
local headless = require("tests.fixture.headless")
local vk = require("vkapi")

local IMG_WIDTH = 4
local IMG_HEIGHT = 4
local PIXEL_SIZE = 4 -- R8G8B8A8 = 4 bytes
local BUFFER_SIZE = IMG_WIDTH * IMG_HEIGHT * PIXEL_SIZE

test.it("should create headless vulkan context", function()
	local ctx = headless.createContext()
	-- Verify context is valid
	test.truthy(ctx.instance)
	test.truthy(ctx.physicalDevice)
	test.truthy(ctx.device)
	test.truthy(ctx.queue)
	test.truthy(ctx.commandPool)
	test.truthy(ctx.memoryProperties)
end)

test.it("should clear to red and read back", function()
	local ctx = headless.createContext()
	local device = ctx.device

	-- Create color attachment (image + view + render pass + framebuffer)
	local image, imgMemory, imageView, renderPass, framebuffer =
		headless.createColorAttachment(ctx, IMG_WIDTH, IMG_HEIGHT, vk.Format.R8G8B8A8_UNORM)

	-- Create staging buffer for readback
	local stagingBuffer, stagingMemory = headless.createStagingBuffer(ctx, BUFFER_SIZE)

	-- Allocate command buffer
	local cmdBuffer = headless.allocateCommandBuffer(ctx)

	-- Record commands: clear to red, then copy to staging buffer
	local beginInfo = vk.CommandBufferBeginInfo({
		flags = vk.CommandBufferUsageFlagBits.ONE_TIME_SUBMIT
	})

	device:beginCommandBuffer(cmdBuffer, beginInfo)

	-- Begin render pass with clear color = red
	local clearValues = vk.ClearValueArray(1)
	clearValues[0].color.float32[0] = 1.0 -- R
	clearValues[0].color.float32[1] = 0.0 -- G
	clearValues[0].color.float32[2] = 0.0 -- B
	clearValues[0].color.float32[3] = 1.0 -- A

	local renderPassBeginInfo = vk.RenderPassBeginInfo({
		renderPass = renderPass,
		framebuffer = framebuffer,
		renderArea = { offset = { x = 0, y = 0 }, extent = { width = IMG_WIDTH, height = IMG_HEIGHT } },
		clearValueCount = 1,
		pClearValues = clearValues
	})

	device:cmdBeginRenderPass(cmdBuffer, renderPassBeginInfo, vk.SubpassContents.INLINE)
	device:cmdEndRenderPass(cmdBuffer)

	-- Pipeline barrier: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
	-- Also provides memory visibility from color attachment writes to transfer reads
	local barriers = vk.ImageMemoryBarrierArray(1)
	barriers[0].srcAccessMask = vk.AccessFlags.COLOR_ATTACHMENT_WRITE
	barriers[0].dstAccessMask = vk.AccessFlags.TRANSFER_READ
	barriers[0].oldLayout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL
	barriers[0].newLayout = vk.ImageLayout.TRANSFER_SRC_OPTIMAL
	barriers[0].srcQueueFamilyIndex = vk.NULL
	barriers[0].dstQueueFamilyIndex = vk.NULL
	barriers[0].image = image
	barriers[0].subresourceRange = vk.ImageSubresourceRange({
		aspectMask = vk.ImageAspectFlagBits.COLOR,
		baseMipLevel = 0,
		levelCount = 1,
		baseArrayLayer = 0,
		layerCount = 1
	})

	device:cmdPipelineBarrier(cmdBuffer,
		vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT,
		vk.PipelineStageFlagBits.TRANSFER,
		1, barriers, 0, nil)

	-- Copy image to staging buffer
	local copyRegions = vk.BufferImageCopyArray(1)
	copyRegions[0].bufferOffset = 0
	copyRegions[0].bufferRowLength = 0
	copyRegions[0].bufferImageHeight = 0
	copyRegions[0].imageSubresource = vk.ImageSubresourceLayers({
		aspectMask = vk.ImageAspectFlagBits.COLOR,
		mipLevel = 0,
		baseArrayLayer = 0,
		layerCount = 1
	})
	copyRegions[0].imageOffset = vk.Offset3D({ x = 0, y = 0, z = 0 })
	copyRegions[0].imageExtent = vk.Extent3D({ width = IMG_WIDTH, height = IMG_HEIGHT, depth = 1 })

	device:cmdCopyImageToBuffer(cmdBuffer, image, vk.ImageLayout.TRANSFER_SRC_OPTIMAL,
		stagingBuffer, 1, copyRegions)

	device:endCommandBuffer(cmdBuffer)

	-- Submit and wait
	headless.submitAndWait(ctx, cmdBuffer)

	-- Read back pixels
	local pixels = headless.readPixels(device, stagingMemory, BUFFER_SIZE)

	-- Verify all pixels are red
	test.equal(#pixels, IMG_WIDTH * IMG_HEIGHT, "should have correct number of pixels")
	for i, px in ipairs(pixels) do
		test.equal(px.r, 255, ("pixel %d R should be 255"):format(i))
		test.equal(px.g, 0, ("pixel %d G should be 0"):format(i))
		test.equal(px.b, 0, ("pixel %d B should be 0"):format(i))
		test.equal(px.a, 255, ("pixel %d A should be 255"):format(i))
	end

	-- Cleanup
	device:destroyFramebuffer(framebuffer)
	device:destroyRenderPass(renderPass)
	device:destroyImageView(imageView)
	device:destroyImage(image)
	device:freeMemory(imgMemory)
	device:destroyBuffer(stagingBuffer)
	device:freeMemory(stagingMemory)
end)

test.it("should clear to green and read back", function()
	local ctx = headless.createContext()
	local device = ctx.device

	local image, imgMemory, imageView, renderPass, framebuffer =
		headless.createColorAttachment(ctx, IMG_WIDTH, IMG_HEIGHT, vk.Format.R8G8B8A8_UNORM)

	local stagingBuffer, stagingMemory = headless.createStagingBuffer(ctx, BUFFER_SIZE)
	local cmdBuffer = headless.allocateCommandBuffer(ctx)

	local beginInfo = vk.CommandBufferBeginInfo({
		flags = vk.CommandBufferUsageFlagBits.ONE_TIME_SUBMIT
	})

	device:beginCommandBuffer(cmdBuffer, beginInfo)

	local clearValues = vk.ClearValueArray(1)
	clearValues[0].color.float32[0] = 0.0 -- R
	clearValues[0].color.float32[1] = 1.0 -- G
	clearValues[0].color.float32[2] = 0.0 -- B
	clearValues[0].color.float32[3] = 1.0 -- A

	local renderPassBeginInfo = vk.RenderPassBeginInfo({
		renderPass = renderPass,
		framebuffer = framebuffer,
		renderArea = { offset = { x = 0, y = 0 }, extent = { width = IMG_WIDTH, height = IMG_HEIGHT } },
		clearValueCount = 1,
		pClearValues = clearValues
	})

	device:cmdBeginRenderPass(cmdBuffer, renderPassBeginInfo, vk.SubpassContents.INLINE)
	device:cmdEndRenderPass(cmdBuffer)

	local barriers = vk.ImageMemoryBarrierArray(1)
	barriers[0].srcAccessMask = vk.AccessFlags.COLOR_ATTACHMENT_WRITE
	barriers[0].dstAccessMask = vk.AccessFlags.TRANSFER_READ
	barriers[0].oldLayout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL
	barriers[0].newLayout = vk.ImageLayout.TRANSFER_SRC_OPTIMAL
	barriers[0].srcQueueFamilyIndex = vk.NULL
	barriers[0].dstQueueFamilyIndex = vk.NULL
	barriers[0].image = image
	barriers[0].subresourceRange = vk.ImageSubresourceRange({
		aspectMask = vk.ImageAspectFlagBits.COLOR,
		baseMipLevel = 0,
		levelCount = 1,
		baseArrayLayer = 0,
		layerCount = 1
	})

	device:cmdPipelineBarrier(cmdBuffer,
		vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT,
		vk.PipelineStageFlagBits.TRANSFER,
		1, barriers, 0, nil)

	local copyRegions = vk.BufferImageCopyArray(1)
	copyRegions[0].bufferOffset = 0
	copyRegions[0].bufferRowLength = 0
	copyRegions[0].bufferImageHeight = 0
	copyRegions[0].imageSubresource = vk.ImageSubresourceLayers({
		aspectMask = vk.ImageAspectFlagBits.COLOR,
		mipLevel = 0,
		baseArrayLayer = 0,
		layerCount = 1
	})
	copyRegions[0].imageOffset = vk.Offset3D({ x = 0, y = 0, z = 0 })
	copyRegions[0].imageExtent = vk.Extent3D({ width = IMG_WIDTH, height = IMG_HEIGHT, depth = 1 })

	device:cmdCopyImageToBuffer(cmdBuffer, image, vk.ImageLayout.TRANSFER_SRC_OPTIMAL,
		stagingBuffer, 1, copyRegions)

	device:endCommandBuffer(cmdBuffer)
	headless.submitAndWait(ctx, cmdBuffer)

	local pixels = headless.readPixels(device, stagingMemory, BUFFER_SIZE)

	test.equal(#pixels, IMG_WIDTH * IMG_HEIGHT)
	for i, px in ipairs(pixels) do
		test.equal(px.r, 0, ("pixel %d R should be 0"):format(i))
		test.equal(px.g, 255, ("pixel %d G should be 255"):format(i))
		test.equal(px.b, 0, ("pixel %d B should be 0"):format(i))
		test.equal(px.a, 255, ("pixel %d A should be 255"):format(i))
	end

	device:destroyFramebuffer(framebuffer)
	device:destroyRenderPass(renderPass)
	device:destroyImageView(imageView)
	device:destroyImage(image)
	device:freeMemory(imgMemory)
	device:destroyBuffer(stagingBuffer)
	device:freeMemory(stagingMemory)
end)
