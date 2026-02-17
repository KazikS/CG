#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (binding = 0, std140) uniform ShadowUniforms {
    mat4 light_view_projection;
} shadow;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float _pad1;
    float shininess;
} object;

void main() {
    gl_Position = shadow.light_view_projection * object.model * vec4(v_position, 1.0);
}
