# vkapi

Cross-platform low level Vulkan bindings for LuaJIT.

## Support

| Arch        | Windows | Linux | macOS |
| ----------- | ------- | ----- | ----- |
| x86-64      | ✅      | ✅    | ❌    |

## Installation

This is intended to be used with [lde](https://lde.sh)

```bash
lde add vkapi --git https://github.com/bycruz/vkapi
```

## Example

You can find an example rendering a triangle in [examples/triangle](examples/triangle)

![Triangle example](examples/triangle/example.png)

## Documentation

Currently there is no documentation, as the api is quite similar, if not identical to the [Vulkan API](https://docs.vulkan.org/spec/latest/index.html), beyond a few changes in the way things are named. Everything is in camelCase, enums are still in SCREAMING_SNAKE_CASE but may have redundant prefixes, ie VK\* or suffixes, ie \_BIT, removed.

A few examples might give you the gist:

`VkCreateInstance` -> `vk.createInstance`  
`VkCreateBuffer` -> `VkDevice:createBuffer`  
`VkQueueFlagBits(VK_QUEUE_GRAPHICS_BIT)` -> `vk.QueueFlagBits.GRAPHICS`

Of course it is 2026 so this is entirely typed with LuaLS annotations.
