# triangle

This example renders a triangle using vkapi.

It uses [winit](https://github.com/bycruz/winit) for window handling, but otherwise no other dependencies.

It also shows best practices for minimizing allocations (putting them outside of the rendering loop, and writing memory in place with LuaJIT) alongside pinning memory to ensure it doesn't get freed.

Additionally, it implements frame buffering to render multiple frames at once for maximum performance.

All in < 500 lines of code, which is pretty good a Vulkan example :P
