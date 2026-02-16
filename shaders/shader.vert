#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

layout(push_constant) uniform PushConstants {
    mat4 projection;
    mat4 transform;
    vec3 color;
} pc;

void main() {
    vec4 pos = vec4(inPosition, 1.0);
    gl_Position = pc.projection * pc.transform * pos;
    vColor = inColor;
}
