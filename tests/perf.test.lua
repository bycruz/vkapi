-- Performance regression test for headless Vulkan operations.
-- Ensures basic round-trip operations stay within reasonable time bounds.

local test = require("lde-test")
local headless = require("tests.fixture.headless")
local vk = require("vkapi")

local IMG_WIDTH = 4
local IMG_HEIGHT = 4
local BUFFER_SIZE = IMG_WIDTH * IMG_HEIGHT * 4
local ITERATIONS = 10

local MAX_TIME_PER_ITER = 0.001

test.it("clear and readback should not regress in performance", function()
	local ctx = headless.createContext()
	local device = ctx.device

	-- Pre-create the reusable resources
	local image, imgMemory, imageView, renderPass, framebuffer =
		headless.createColorAttachment(ctx, IMG_WIDTH, IMG_HEIGHT, vk.Format.R8G8B8A8_UNORM)

	local stagingBuffer, stagingMemory = headless.createStagingBuffer(ctx, BUFFER_SIZE)

	-- Record the template command buffer once, then reuse it
	-- (we can reset + re-record, but for perf testing we measure end-to-end)
	local function runClearIteration(r, g, b, a)
		local cmdBuffer = headless.allocateCommandBuffer(ctx)

		local beginInfo = vk.CommandBufferBeginInfo({
			flags = vk.CommandBufferUsageFlagBits.ONE_TIME_SUBMIT
		})
		device:beginCommandBuffer(cmdBuffer, beginInfo)

		local clearValues = vk.ClearValueArray(1)
		clearValues[0].color.float32[0] = r
		clearValues[0].color.float32[1] = g
		clearValues[0].color.float32[2] = b
		clearValues[0].color.float32[3] = a

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
		headless.readPixels(device, stagingMemory, BUFFER_SIZE)
	end

	-- Warmup run (compilation overhead, driver init)
	runClearIteration(1, 0, 0, 1)

	-- Timed runs
	local start = os.clock()
	for i = 1, ITERATIONS do
		-- Alternate colors so each iteration is a fresh command buffer
		local color = i % 2 == 0 and { 1, 0, 0, 1 } or { 0, 1, 0, 1 }
		runClearIteration(color[1], color[2], color[3], color[4])
	end
	local elapsed = os.clock() - start

	local avgTime = elapsed / ITERATIONS
	test.less(avgTime, MAX_TIME_PER_ITER,
		("average iteration time %.3fs exceeds limit of %.1fs"):format(avgTime, MAX_TIME_PER_ITER))

	-- Cleanup
	device:destroyFramebuffer(framebuffer)
	device:destroyRenderPass(renderPass)
	device:destroyImageView(imageView)
	device:destroyImage(image)
	device:freeMemory(imgMemory)
	device:destroyBuffer(stagingBuffer)
	device:freeMemory(stagingMemory)
end)
