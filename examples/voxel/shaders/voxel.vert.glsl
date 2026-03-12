#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = pc.mvp * vec4(aPos, 1.0);

    vec3 lightDir = normalize(vec3(0.8, 1.0, 0.6));
    float diffuse = max(dot(aNormal, lightDir), 0.0);
    float ambient = 0.35;
    float light = ambient + diffuse * 0.65;

    fragColor = aColor * light;
}
