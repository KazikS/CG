#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

// ========== Data structures ==========

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    float ambient_intensity;
    uint32_t point_light_count;
    uint32_t spotlight_count;
    float _pad1;

    veekay::vec3 light_direction;
    float _pad3;
    veekay::vec3 light_color;
    float light_intensity;

    veekay::vec3 camera_position;
    float _pad4;
};

struct Spotlight {
    veekay::vec3 position;
    float _pad0;
    veekay::vec3 direction;
    float _pad1;
    veekay::vec3 color;
    float intensity;
    float inner_cutoff;
    float outer_cutoff;
    float _pad2, _pad3;
};

struct ModelUniforms {
    veekay::mat4 model;
    veekay::vec3 albedo_color; float _pad0;
    veekay::vec3 specular_color; float _pad1;
    float shininess; float _pad2, _pad3, _pad4;
};

struct Mesh {
    veekay::graphics::Buffer* vertex_buffer;
    veekay::graphics::Buffer* index_buffer;
    uint32_t indices;
};

struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};

    veekay::mat4 matrix() const;
};

struct Model {
    Mesh mesh;
    Transform transform;
    veekay::vec3 albedo_color;
    veekay::vec3 specular_color;
    float shininess;
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    veekay::vec3 position = {};
    veekay::vec3 rotation = {};

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    veekay::mat4 view() const;
    veekay::mat4 view_projection(float aspect_ratio) const;
};

// ========== Scene objects ==========

inline namespace {
    Camera camera{
        .position = {0.0f, -2.0f, 8.0f},
    };

    std::vector<Model> models;
    std::vector<Spotlight> spotlights;
}

// ========== Vulkan objects ==========

inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* spotlights_buffer;

    Mesh plane_mesh;
    Mesh cube_mesh;
    Mesh pyramid_mesh;
    Mesh sphere_mesh;

    // Texture objects
    veekay::graphics::Texture* albedo_texture;
    VkSampler texture_sampler;
}

// ========== Helpers ==========

float toRadians(float degrees) {
    return degrees * float(M_PI) / 180.0f;
}

template<typename T>
T clampVal(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

veekay::mat4 Transform::matrix() const {
    veekay::mat4 trans = veekay::mat4::translation(position);
    veekay::mat4 rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
    veekay::mat4 rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
    veekay::mat4 rot_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
    veekay::mat4 scale_mat = veekay::mat4::scaling(scale);

    return trans * rot_z * rot_y * rot_x * scale_mat;
}

veekay::mat4 Camera::view() const {
    auto t = veekay::mat4::translation(-position);
    auto rx = veekay::mat4::rotation({1, 0, 0}, toRadians(-rotation.x));
    auto ry = veekay::mat4::rotation({0, 1, 0}, toRadians(-rotation.y - 180.0f));
    auto rz = veekay::mat4::rotation({0, 0, 1}, toRadians(-rotation.z));
    return t * rz * ry * rx;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
    auto proj = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    return view() * proj;
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return VK_NULL_HANDLE;
    size_t size = static_cast<size_t>(file.tellg());
    if (size == 0) { file.close(); return VK_NULL_HANDLE; }
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return result;
}

// Load texture from PNG file
veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const char* path) {
    std::vector<unsigned char> image;
    unsigned width, height;

    unsigned error = lodepng::decode(image, width, height, path);
    if (error) {
        std::cerr << "Failed to load texture: " << path << " (" << lodepng_error_text(error) << ")\n";
        return nullptr;
    }

    std::cout << "Loaded texture: " << path << " (" << width << "x" << height << ")\n";
    return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image.data());
}

// ========== Geometry generation ==========

Mesh createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    Mesh mesh;
    mesh.vertex_buffer = new veekay::graphics::Buffer(
        vertices.size() * sizeof(Vertex), vertices.data(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.index_buffer = new veekay::graphics::Buffer(
        indices.size() * sizeof(uint32_t), indices.data(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.indices = static_cast<uint32_t>(indices.size());
    return mesh;
}

Mesh generatePlane() {
    std::vector<Vertex> vertices = {
        {{-5.0f, 0.0f, 5.0f},  {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 5.0f, 0.0f, 5.0f},  {0.0f, -1.0f, 0.0f}, {5.0f, 0.0f}},
        {{ 5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {5.0f, 5.0f}},
        {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 5.0f}},
    };
    std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
    return createMesh(vertices, indices);
}

Mesh generateCube() {
    std::vector<Vertex> vertices = {
        // Front
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0, 0}},
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1, 0}},
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1, 1}},
        {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0, 1}},
        // Right
        {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0, 0}},
        {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1, 0}},
        {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1, 1}},
        {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0, 1}},
        // Back
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0, 0}},
        {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1, 0}},
        {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1, 1}},
        {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0, 1}},
        // Left
        {{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {0, 0}},
        {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1, 0}},
        {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1, 1}},
        {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {0, 1}},
        // Bottom
        {{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0, 0}},
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1, 0}},
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1, 1}},
        {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0, 1}},
        // Top
        {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0, 0}},
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1, 0}},
        {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1, 1}},
        {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0, 1}},
    };
    std::vector<uint32_t> indices = {
        0,1,2, 2,3,0,  4,5,6, 6,7,4,  8,9,10, 10,11,8,
        12,13,14, 14,15,12,  16,17,18, 18,19,16,  20,21,22, 22,23,20,
    };
    return createMesh(vertices, indices);
}

Mesh generatePyramid() {
    veekay::vec3 apex = {0.0f, 0.7f, 0.0f};
    veekay::vec3 b0 = {-0.5f, -0.3f, -0.5f};
    veekay::vec3 b1 = { 0.5f, -0.3f, -0.5f};
    veekay::vec3 b2 = { 0.5f, -0.3f,  0.5f};
    veekay::vec3 b3 = {-0.5f, -0.3f,  0.5f};

    auto faceNormal = [](veekay::vec3 a, veekay::vec3 b, veekay::vec3 c) {
        veekay::vec3 e1 = b - a, e2 = c - a;
        return veekay::vec3::normalized(veekay::vec3::cross(e1, e2));
    };

    veekay::vec3 n_front = faceNormal(b0, b1, apex);
    veekay::vec3 n_right = faceNormal(b1, b2, apex);
    veekay::vec3 n_back  = faceNormal(b2, b3, apex);
    veekay::vec3 n_left  = faceNormal(b3, b0, apex);
    veekay::vec3 n_bot   = {0.0f, -1.0f, 0.0f};

    std::vector<Vertex> vertices = {
        {b0, n_front, {0,0}}, {b1, n_front, {1,0}}, {apex, n_front, {0.5f,1}},
        {b1, n_right, {0,0}}, {b2, n_right, {1,0}}, {apex, n_right, {0.5f,1}},
        {b2, n_back, {0,0}},  {b3, n_back, {1,0}},  {apex, n_back, {0.5f,1}},
        {b3, n_left, {0,0}},  {b0, n_left, {1,0}},  {apex, n_left, {0.5f,1}},
        {b0, n_bot, {0,0}}, {b1, n_bot, {1,0}}, {b2, n_bot, {1,1}}, {b3, n_bot, {0,1}},
    };
    std::vector<uint32_t> indices = {
        0,1,2, 3,4,5, 6,7,8, 9,10,11,
        12,13,14, 14,15,12,
    };
    return createMesh(vertices, indices);
}

Mesh generateSphere(int stacks, int slices) {
    const float radius = 0.5f;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (int i = 0; i <= stacks; ++i) {
        float v = float(i) / float(stacks);
        float phi = v * float(M_PI);
        float sinPhi = sinf(phi), cosPhi = cosf(phi);
        for (int j = 0; j <= slices; ++j) {
            float u = float(j) / float(slices);
            float theta = u * 2.0f * float(M_PI);
            float x = sinPhi * cosf(theta);
            float y = cosPhi;
            float z = sinPhi * sinf(theta);
            Vertex vert;
            vert.position = {x * radius, y * radius, z * radius};
            vert.normal = {x, y, z};
            vert.uv = {u, v};
            vertices.push_back(vert);
        }
    }
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t first = uint32_t(i * (slices + 1) + j);
            uint32_t second = first + uint32_t(slices + 1);
            indices.push_back(first);     indices.push_back(second);     indices.push_back(first + 1);
            indices.push_back(second);    indices.push_back(second + 1); indices.push_back(first + 1);
        }
    }
    return createMesh(vertices, indices);
}

// ========== Application callbacks ==========

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;

    // Load shaders
    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    if (!vertex_shader_module) {
        std::cerr << "Failed to load vertex shader\n";
        veekay::app.running = false; return;
    }
    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
    if (!fragment_shader_module) {
        std::cerr << "Failed to load fragment shader\n";
        veekay::app.running = false; return;
    }

    // Load texture from file
    albedo_texture = loadTexture(cmd, "assets/texture.png");
    if (!albedo_texture) {
        std::cerr << "Failed to load texture, creating fallback\n";
        // Create a simple white 2x2 fallback texture
        uint32_t white_pixels[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        albedo_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, white_pixels);
    }

    // Create sampler with reasonable parameters
    {
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
        };
        if (vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create texture sampler\n";
            veekay::app.running = false; return;
        }
    }

    // Build pipeline
    {
        VkPipelineShaderStageCreateInfo stage_infos[2];
        stage_infos[0] = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };
        stage_infos[1] = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        VkVertexInputBindingDescription buffer_binding{
            .binding = 0, .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };
        VkVertexInputAttributeDescription attributes[] = {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)},
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv)},
        };
        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes,
        };
        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = false, .minSampleShading = 1.0f,
        };
        VkViewport viewport{
            .x = 0, .y = 0,
            .width = float(veekay::app.window_width),
            .height = float(veekay::app.window_height),
            .minDepth = 0, .maxDepth = 1,
        };
        VkRect2D scissor{.offset = {0,0}, .extent = {veekay::app.window_width, veekay::app.window_height}};
        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport,
            .scissorCount = 1, .pScissors = &scissor,
        };
        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = true, .depthWriteEnable = true,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };
        VkPipelineColorBlendAttachmentState attachment_info{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false, .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1, .pAttachments = &attachment_info,
        };

        // Descriptor pool
        VkDescriptorPoolSize pools[] = {
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 4},
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 4},
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4},
            {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 4},
        };
        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 4,
            .poolSizeCount = 4, .pPoolSizes = pools,
        };
        vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool);

        // Descriptor set layout: binding 0 = scene UBO, 1 = model UBO dynamic, 2 = spotlights SSBO, 3 = texture sampler
        VkDescriptorSetLayoutBinding bindings[] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
        };
        VkDescriptorSetLayoutCreateInfo layout_ci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4, .pBindings = bindings,
        };
        vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_set_layout);

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout,
        };
        vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set);

        // Pipeline layout
        VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout,
        };
        vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);

        // Pipeline
        VkGraphicsPipelineCreateInfo pipeline_ci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline);
    }

    // Create uniform/storage buffers
    scene_uniforms_buffer = new veekay::graphics::Buffer(
        sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    model_uniforms_buffer = new veekay::graphics::Buffer(
        max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
        nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    spotlights_buffer = new veekay::graphics::Buffer(
        8 * sizeof(Spotlight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Write descriptor set
    {
        VkDescriptorBufferInfo buffer_infos[] = {
            {.buffer = scene_uniforms_buffer->buffer, .offset = 0, .range = sizeof(SceneUniforms)},
            {.buffer = model_uniforms_buffer->buffer, .offset = 0, .range = sizeof(ModelUniforms)},
            {.buffer = spotlights_buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        };
        VkDescriptorImageInfo image_info{
            .sampler = texture_sampler,
            .imageView = albedo_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet writes[] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
             .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_infos[0]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
             .dstBinding = 1, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffer_infos[1]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
             .dstBinding = 2, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_infos[2]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
             .dstBinding = 3, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_info},
        };
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    // Generate meshes
    plane_mesh   = generatePlane();
    cube_mesh    = generateCube();
    pyramid_mesh = generatePyramid();
    sphere_mesh  = generateSphere(20, 20);

    // Add models to scene
    // Floor - white albedo so texture shows its true colors
    models.push_back({
        .mesh = plane_mesh,
        .transform = Transform{},
        .albedo_color = {1.0f, 1.0f, 1.0f},
        .specular_color = {0.2f, 0.2f, 0.2f},
        .shininess = 4.0f,
    });

    // Blue cube
    models.push_back({
        .mesh = cube_mesh,
        .transform = Transform{.position = {-2.5f, -0.5f, 0.0f}},
        .albedo_color = {0.5f, 0.7f, 1.0f},
        .specular_color = {1.0f, 1.0f, 1.0f},
        .shininess = 32.0f,
    });

    // Red pyramid
    models.push_back({
        .mesh = pyramid_mesh,
        .transform = Transform{.position = {0.0f, -0.3f, 0.0f}, .scale = {1.2f, 1.2f, 1.2f}},
        .albedo_color = {1.0f, 0.5f, 0.4f},
        .specular_color = {1.0f, 0.8f, 0.8f},
        .shininess = 16.0f,
    });

    // Green sphere
    models.push_back({
        .mesh = sphere_mesh,
        .transform = Transform{.position = {2.5f, -0.5f, 0.0f}},
        .albedo_color = {0.5f, 1.0f, 0.6f},
        .specular_color = {1.0f, 1.0f, 1.0f},
        .shininess = 64.0f,
    });

    // White cube in the back
    models.push_back({
        .mesh = cube_mesh,
        .transform = Transform{.position = {0.0f, -0.5f, -3.0f}, .scale = {1.5f, 1.5f, 1.5f}},
        .albedo_color = {1.0f, 1.0f, 1.0f},
        .specular_color = {1.0f, 1.0f, 1.0f},
        .shininess = 8.0f,
    });

    // Initial spotlights
    spotlights.push_back(Spotlight{
        .position = {0.0f, -4.0f, 2.0f},
        .direction = {0.0f, 1.0f, -0.3f},
        .color = {1.0f, 0.95f, 0.8f},
        .intensity = 60.0f,
        .inner_cutoff = std::cos(toRadians(15.0f)),
        .outer_cutoff = std::cos(toRadians(25.0f)),
    });

    spotlights.push_back(Spotlight{
        .position = {3.0f, -3.0f, 3.0f},
        .direction = {-0.5f, 0.7f, -0.5f},
        .color = {0.3f, 0.5f, 1.0f},
        .intensity = 40.0f,
        .inner_cutoff = std::cos(toRadians(12.0f)),
        .outer_cutoff = std::cos(toRadians(20.0f)),
    });
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    delete sphere_mesh.index_buffer;  delete sphere_mesh.vertex_buffer;
    delete pyramid_mesh.index_buffer; delete pyramid_mesh.vertex_buffer;
    delete cube_mesh.index_buffer;    delete cube_mesh.vertex_buffer;
    delete plane_mesh.index_buffer;   delete plane_mesh.vertex_buffer;

    vkDestroySampler(device, texture_sampler, nullptr);
    delete albedo_texture;

    delete spotlights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;

    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    (void)time;

    ImGui::Begin("Lighting Controls");

    ImGui::Text("Camera (Transform mode)");
    ImGui::Text("Position: (%.1f, %.1f, %.1f)", camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("WASD - move, Mouse LMB - look");
    ImGui::Separator();

    // Ambient
    static float ambient_intensity = 0.1f;
    ImGui::Text("Ambient Light");
    ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
    ImGui::Separator();

    // Directional
    static float dir_light_dir[3] = {0.3f, -1.0f, 0.2f};
    static float dir_light_color[3] = {1.0f, 1.0f, 0.9f};
    static float dir_light_intensity = 1.5f;

    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", dir_light_dir, -1.0f, 1.0f);
    ImGui::ColorEdit3("Color##dir", dir_light_color);
    ImGui::SliderFloat("Intensity##dir", &dir_light_intensity, 0.0f, 5.0f);
    ImGui::Separator();

    // Spotlights
    ImGui::Text("Spotlights");

    static std::vector<std::pair<float, float>> spotlight_angles;
    if (spotlight_angles.size() != spotlights.size()) {
        spotlight_angles.resize(spotlights.size());
        for (size_t i = 0; i < spotlights.size(); ++i) {
            float ic = clampVal(spotlights[i].inner_cutoff, 0.0f, 1.0f);
            float oc = clampVal(spotlights[i].outer_cutoff, 0.0f, 1.0f);
            spotlight_angles[i].first  = std::acos(ic) * 180.0f / float(M_PI);
            spotlight_angles[i].second = std::acos(oc) * 180.0f / float(M_PI);
        }
    }

    for (size_t i = 0; i < spotlights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::TreeNode("Spotlight", "Spotlight %zu", i)) {
            ImGui::SliderFloat3("Position", &spotlights[i].position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &spotlights[i].direction.x, -1.0f, 1.0f);
            spotlights[i].direction = veekay::vec3::normalized(spotlights[i].direction);
            ImGui::ColorEdit3("Color", &spotlights[i].color.x);
            ImGui::SliderFloat("Intensity", &spotlights[i].intensity, 0.0f, 200.0f);

            float& inner_angle = spotlight_angles[i].first;
            float& outer_angle = spotlight_angles[i].second;
            if (ImGui::SliderFloat("Inner Angle", &inner_angle, 0.0f, 45.0f)) {
                spotlights[i].inner_cutoff = std::cos(toRadians(inner_angle));
                if (outer_angle < inner_angle + 1.0f) {
                    outer_angle = inner_angle + 1.0f;
                    spotlights[i].outer_cutoff = std::cos(toRadians(outer_angle));
                }
            }
            if (ImGui::SliderFloat("Outer Angle", &outer_angle, inner_angle + 1.0f, 45.0f)) {
                spotlights[i].outer_cutoff = std::cos(toRadians(outer_angle));
            }
            if (ImGui::Button("Remove")) {
                spotlights.erase(spotlights.begin() + i);
                spotlight_angles.erase(spotlight_angles.begin() + i);
                ImGui::TreePop(); ImGui::PopID(); ImGui::End();
                return;
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (spotlights.size() < 8 && ImGui::Button("Add Spotlight")) {
        spotlights.push_back(Spotlight{
            .position = {0.0f, -3.0f, 0.0f},
            .direction = {0.0f, 1.0f, 0.0f},
            .color = {1.0f, 1.0f, 1.0f},
            .intensity = 50.0f,
            .inner_cutoff = std::cos(toRadians(12.5f)),
            .outer_cutoff = std::cos(toRadians(17.5f)),
        });
        spotlight_angles.push_back({12.5f, 17.5f});
    }

    ImGui::End();

    // Camera control
    if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        using namespace veekay::input;

        if (mouse::isButtonDown(mouse::Button::left)) {
            auto move_delta = mouse::cursorDelta();
            camera.rotation.y += move_delta.x * 0.2f;
            camera.rotation.x -= move_delta.y * 0.2f;
            if (camera.rotation.x > 89.0f)  camera.rotation.x = 89.0f;
            if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
        }

        auto v = camera.view();
        veekay::vec3 right = {v.elements[0][0], v.elements[1][0], v.elements[2][0]};
        veekay::vec3 up    = {v.elements[0][1], v.elements[1][1], v.elements[2][1]};
        veekay::vec3 front = {v.elements[0][2], v.elements[1][2], v.elements[2][2]};

        float speed = 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::w)) camera.position += front * speed;
        if (keyboard::isKeyDown(keyboard::Key::s)) camera.position -= front * speed;
        if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * speed;
        if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * speed;
        if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * speed;
        if (keyboard::isKeyDown(keyboard::Key::e)) camera.position += up * speed;
    }

    // Update uniform buffers
    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .ambient_intensity = ambient_intensity,
        .point_light_count = 0,
        .spotlight_count = static_cast<uint32_t>(spotlights.size()),
        .light_direction = veekay::vec3::normalized({dir_light_dir[0], dir_light_dir[1], dir_light_dir[2]}),
        .light_color = {dir_light_color[0], dir_light_color[1], dir_light_color[2]},
        .light_intensity = dir_light_intensity,
        .camera_position = camera.position,
    };

    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
    for (size_t i = 0; i < models.size(); ++i) {
        const Model& model = models[i];
        ModelUniforms uniforms;
        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.albedo_color;
        uniforms.specular_color = model.specular_color;
        uniforms.shininess = model.shininess;

        char* ptr = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(ptr) = uniforms;
    }

    if (!spotlights.empty()) {
        std::memcpy(spotlights_buffer->mapped_region, spotlights.data(),
                    spotlights.size() * sizeof(Spotlight));
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);

    {
        VkCommandBufferBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &info);
    }

    {
        VkClearValue clear_color{.color = {{0.05f, 0.05f, 0.08f, 1.0f}}};
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        VkClearValue clear_values[] = {clear_color, clear_depth};
        VkRenderPassBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = veekay::app.vk_render_pass,
            .framebuffer = framebuffer,
            .renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
            .clearValueCount = 2, .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize zero_offset = 0;

    const size_t model_alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    for (size_t i = 0; i < models.size(); ++i) {
        const Mesh& mesh = models[i].mesh;

        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertex_buffer->buffer, &zero_offset);
        vkCmdBindIndexBuffer(cmd, mesh.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);

        uint32_t offset = static_cast<uint32_t>(i * model_alignment);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                0, 1, &descriptor_set, 1, &offset);

        vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}
