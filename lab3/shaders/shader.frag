#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    float ambient_intensity;
    uint point_light_count;
    uint spotlight_count;
    float _pad1;
    
    vec3 light_direction;
    float _pad3;
    vec3 light_color;
    float light_intensity;
        
    vec3 camera_position;
    float _pad4;
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float _pad1;
    float shininess;
} object;

struct Spotlight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float intensity;
    float inner_cutoff;
    float outer_cutoff;
    float _pad2, _pad3;
};

layout (binding = 2, std430) readonly buffer SpotLights {
    Spotlight lights[];
} spot_lights;

layout (binding = 3) uniform sampler2D albedo_texture;

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(scene.camera_position - f_position);

    // Sample texture and multiply by material albedo color
    vec4 tex_color = texture(albedo_texture, f_uv);
    vec3 albedo = tex_color.rgb * object.albedo_color;

    // Ambient
    vec3 ambient = scene.ambient_intensity * albedo;

    // Directional light (Blinn-Phong)
    float diff = max(dot(normal, scene.light_direction), 0.0);
    vec3 diffuse = diff * scene.light_intensity * scene.light_color * albedo;
    
    vec3 halfway_dir = normalize(scene.light_direction + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), object.shininess);
    vec3 specular = spec * scene.light_intensity * scene.light_color * object.specular_color;

    // Spotlights
    vec3 spotlight_lighting = vec3(0.0);
    for (uint i = 0; i < scene.spotlight_count; ++i) {
        Spotlight light = spot_lights.lights[i];
        
        vec3 light_dir = light.position - f_position;
        float distance = length(light_dir);
        light_dir = normalize(light_dir);
        
        float theta = dot(light_dir, normalize(-light.direction));
        float epsilon = light.inner_cutoff - light.outer_cutoff;
        float spot_intensity = clamp((theta - light.outer_cutoff) / epsilon, 0.0, 1.0);
        
        float attenuation = light.intensity / (distance * distance + 0.01);
        float total_attenuation = attenuation * spot_intensity;
        
        float spot_diff = max(dot(normal, light_dir), 0.0);
        vec3 spot_diffuse = spot_diff * total_attenuation * light.color * albedo;
        
        vec3 spot_halfway = normalize(light_dir + view_dir);
        float spot_spec = pow(max(dot(normal, spot_halfway), 0.0), object.shininess);
        vec3 spot_specular = spot_spec * total_attenuation * light.color * object.specular_color;
        
        spotlight_lighting += spot_diffuse + spot_specular;
    }

    vec3 result = ambient + diffuse + specular + spotlight_lighting;
    
    final_color = vec4(result, 1.0);
}
