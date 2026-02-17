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
constexpr uint32_t shadow_map_size = 2048;

// ========== Dynamic Rendering function pointers ==========
PFN_vkCmdBeginRenderingKHR pfnCmdBeginRendering = nullptr;
PFN_vkCmdEndRenderingKHR pfnCmdEndRendering = nullptr;

// ========== Data structures ==========

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::mat4 light_view_projection;
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

struct ShadowUniforms {
    veekay::mat4 light_view_projection;
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
    Camera camera{.position = {0.0f, -2.0f, 8.0f}};
    std::vector<Model> models;
    std::vector<Spotlight> spotlights;
}

// ========== Vulkan objects ==========
inline namespace {
    // Main pipeline
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    // Shadow pipeline
    VkShaderModule shadow_vert_module;
    VkShaderModule shadow_frag_module;
    VkDescriptorSetLayout shadow_descriptor_set_layout;
    VkDescriptorSet shadow_descriptor_set;
    VkPipelineLayout shadow_pipeline_layout;
    VkPipeline shadow_pipeline;

    // Buffers
    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* spotlights_buffer;
    veekay::graphics::Buffer* shadow_uniforms_buffer;

    // Meshes
    Mesh plane_mesh, cube_mesh, pyramid_mesh, sphere_mesh;

    // Texture
    veekay::graphics::Texture* albedo_texture;
    VkSampler texture_sampler;

    // Shadow map
    VkImage shadow_image;
    VkDeviceMemory shadow_image_memory;
    VkImageView shadow_image_view;
    VkSampler shadow_sampler;
}

// ========== Helpers ==========

float toRadians(float degrees) { return degrees * float(M_PI) / 180.0f; }

template<typename T>
T clampVal(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

veekay::mat4 Transform::matrix() const {
    auto trans = veekay::mat4::translation(position);
    auto rx = veekay::mat4::rotation({1,0,0}, toRadians(rotation.x));
    auto ry = veekay::mat4::rotation({0,1,0}, toRadians(rotation.y));
    auto rz = veekay::mat4::rotation({0,0,1}, toRadians(rotation.z));
    auto sc = veekay::mat4::scaling(scale);
    return trans * rz * ry * rx * sc;
}

veekay::mat4 Camera::view() const {
    auto t = veekay::mat4::translation(-position);
    auto rx = veekay::mat4::rotation({1,0,0}, toRadians(-rotation.x));
    auto ry = veekay::mat4::rotation({0,1,0}, toRadians(-rotation.y - 180.0f));
    auto rz = veekay::mat4::rotation({0,0,1}, toRadians(-rotation.z));
    return t * rz * ry * rx;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
    return view() * veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
}

// Orthographic projection for directional light shadow
veekay::mat4 orthoProjection(float left, float right, float bottom, float top, float near_p, float far_p) {
    veekay::mat4 r{};
    r.elements[0][0] = 2.0f / (right - left);
    r.elements[1][1] = 2.0f / (top - bottom);
    r.elements[2][2] = 1.0f / (far_p - near_p);
    r.elements[3][0] = -(right + left) / (right - left);
    r.elements[3][1] = -(top + bottom) / (top - bottom);
    r.elements[3][2] = -near_p / (far_p - near_p);
    r.elements[3][3] = 1.0f;
    return r;
}

// Look-at matrix for light view
veekay::mat4 lookAt(veekay::vec3 eye, veekay::vec3 target, veekay::vec3 world_up) {
    veekay::vec3 forward = veekay::vec3::normalized(target - eye);
    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, world_up));
    veekay::vec3 up = veekay::vec3::cross(right, forward);

    veekay::mat4 r = veekay::mat4::identity();
    r.elements[0][0] = right.x;   r.elements[1][0] = right.y;   r.elements[2][0] = right.z;
    r.elements[0][1] = up.x;      r.elements[1][1] = up.y;      r.elements[2][1] = up.z;
    r.elements[0][2] = -forward.x; r.elements[1][2] = -forward.y; r.elements[2][2] = -forward.z;
    r.elements[3][0] = -veekay::vec3::dot(right, eye);
    r.elements[3][1] = -veekay::vec3::dot(up, eye);
    r.elements[3][2] = veekay::vec3::dot(forward, eye);
    return r;
}

veekay::mat4 computeLightVP(veekay::vec3 light_dir) {
    veekay::vec3 light_pos = light_dir * -15.0f;
    veekay::vec3 target = {0, 0, 0};
    veekay::vec3 up = {0, -1, 0};
    if (std::abs(veekay::vec3::dot(veekay::vec3::normalized(light_dir), veekay::vec3{0,-1,0})) > 0.99f)
        up = {0, 0, 1};
    veekay::mat4 view = lookAt(light_pos, target, up);
    veekay::mat4 proj = orthoProjection(-10, 10, -10, 10, 0.1f, 40.0f);
    return view * proj;
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
        .codeSize = size, .pCode = buffer.data(),
    };
    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return result;
}

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

// ========== Geometry ==========

Mesh createMesh(const std::vector<Vertex>& v, const std::vector<uint32_t>& idx) {
    Mesh m;
    m.vertex_buffer = new veekay::graphics::Buffer(v.size()*sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m.index_buffer = new veekay::graphics::Buffer(idx.size()*sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m.indices = (uint32_t)idx.size();
    return m;
}

Mesh generatePlane() {
    std::vector<Vertex> v = {
        {{-5,0,5},{0,-1,0},{0,0}}, {{5,0,5},{0,-1,0},{5,0}},
        {{5,0,-5},{0,-1,0},{5,5}}, {{-5,0,-5},{0,-1,0},{0,5}},
    };
    return createMesh(v, {0,1,2,2,3,0});
}

Mesh generateCube() {
    std::vector<Vertex> v = {
        {{-.5f,-.5f,-.5f},{0,0,-1},{0,0}},{{.5f,-.5f,-.5f},{0,0,-1},{1,0}},{{.5f,.5f,-.5f},{0,0,-1},{1,1}},{{-.5f,.5f,-.5f},{0,0,-1},{0,1}},
        {{.5f,-.5f,-.5f},{1,0,0},{0,0}},{{.5f,-.5f,.5f},{1,0,0},{1,0}},{{.5f,.5f,.5f},{1,0,0},{1,1}},{{.5f,.5f,-.5f},{1,0,0},{0,1}},
        {{.5f,-.5f,.5f},{0,0,1},{0,0}},{{-.5f,-.5f,.5f},{0,0,1},{1,0}},{{-.5f,.5f,.5f},{0,0,1},{1,1}},{{.5f,.5f,.5f},{0,0,1},{0,1}},
        {{-.5f,-.5f,.5f},{-1,0,0},{0,0}},{{-.5f,-.5f,-.5f},{-1,0,0},{1,0}},{{-.5f,.5f,-.5f},{-1,0,0},{1,1}},{{-.5f,.5f,.5f},{-1,0,0},{0,1}},
        {{-.5f,-.5f,.5f},{0,-1,0},{0,0}},{{.5f,-.5f,.5f},{0,-1,0},{1,0}},{{.5f,-.5f,-.5f},{0,-1,0},{1,1}},{{-.5f,-.5f,-.5f},{0,-1,0},{0,1}},
        {{-.5f,.5f,-.5f},{0,1,0},{0,0}},{{.5f,.5f,-.5f},{0,1,0},{1,0}},{{.5f,.5f,.5f},{0,1,0},{1,1}},{{-.5f,.5f,.5f},{0,1,0},{0,1}},
    };
    return createMesh(v, {0,1,2,2,3,0,4,5,6,6,7,4,8,9,10,10,11,8,12,13,14,14,15,12,16,17,18,18,19,16,20,21,22,22,23,20});
}

Mesh generatePyramid() {
    veekay::vec3 a={0,.7f,0},b0={-.5f,-.3f,-.5f},b1={.5f,-.3f,-.5f},b2={.5f,-.3f,.5f},b3={-.5f,-.3f,.5f};
    auto fn=[](veekay::vec3 a,veekay::vec3 b,veekay::vec3 c){
        auto e1=b-a,e2=c-a;return veekay::vec3::normalized(veekay::vec3::cross(e1,e2));};
    auto nf=fn(b0,b1,a),nr=fn(b1,b2,a),nb=fn(b2,b3,a),nl=fn(b3,b0,a);
    veekay::vec3 nd={0,-1,0};
    std::vector<Vertex> v={
        {b0,nf,{0,0}},{b1,nf,{1,0}},{a,nf,{.5f,1}},
        {b1,nr,{0,0}},{b2,nr,{1,0}},{a,nr,{.5f,1}},
        {b2,nb,{0,0}},{b3,nb,{1,0}},{a,nb,{.5f,1}},
        {b3,nl,{0,0}},{b0,nl,{1,0}},{a,nl,{.5f,1}},
        {b0,nd,{0,0}},{b1,nd,{1,0}},{b2,nd,{1,1}},{b3,nd,{0,1}},
    };
    return createMesh(v, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,14,15,12});
}

Mesh generateSphere(int st, int sl) {
    std::vector<Vertex> v; std::vector<uint32_t> idx;
    for(int i=0;i<=st;++i){float vv=float(i)/st;float phi=vv*float(M_PI);
        for(int j=0;j<=sl;++j){float u=float(j)/sl;float th=u*2*float(M_PI);
            float x=sinf(phi)*cosf(th),y=cosf(phi),z=sinf(phi)*sinf(th);
            v.push_back({{x*.5f,y*.5f,z*.5f},{x,y,z},{u,vv}});}}
    for(int i=0;i<st;++i)for(int j=0;j<sl;++j){
        uint32_t f=i*(sl+1)+j,s=f+sl+1;
        idx.push_back(f);idx.push_back(s);idx.push_back(f+1);
        idx.push_back(s);idx.push_back(s+1);idx.push_back(f+1);}
    return createMesh(v,idx);
}

// ========== Shadow map creation ==========

uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++)
        if ((type_filter & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}

void createShadowMap() {
    VkDevice& device = veekay::app.vk_device;

    // Create depth image
    VkImageCreateInfo img_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .extent = {shadow_map_size, shadow_map_size, 1},
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    vkCreateImage(device, &img_info, nullptr, &shadow_image);

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device, shadow_image, &mem_req);
    VkMemoryAllocateInfo alloc{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    vkAllocateMemory(device, &alloc, nullptr, &shadow_image_memory);
    vkBindImageMemory(device, shadow_image, shadow_image_memory, 0);

    // Create image view
    VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = shadow_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vkCreateImageView(device, &view_info, nullptr, &shadow_image_view);

    // Create shadow sampler with comparison
    VkSamplerCreateInfo samp_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };
    vkCreateSampler(device, &samp_info, nullptr, &shadow_sampler);
}

// ========== Application callbacks ==========

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;

    // Load dynamic rendering extension functions
    pfnCmdBeginRendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR");
    pfnCmdEndRendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR");
    if (!pfnCmdBeginRendering || !pfnCmdEndRendering) {
        std::cerr << "Failed to load Dynamic Rendering functions\n";
        veekay::app.running = false; return;
    }

    // Load shaders
    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
    shadow_vert_module = loadShaderModule("./shaders/shadow.vert.spv");
    shadow_frag_module = loadShaderModule("./shaders/shadow.frag.spv");
    if (!vertex_shader_module || !fragment_shader_module || !shadow_vert_module || !shadow_frag_module) {
        std::cerr << "Failed to load shaders\n";
        veekay::app.running = false; return;
    }

    // Load texture
    albedo_texture = loadTexture(cmd, "assets/texture.png");
    if (!albedo_texture) {
        uint32_t white[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        albedo_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, white);
    }

    VkSamplerCreateInfo tex_samp_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_TRUE, .maxAnisotropy = 16.0f,
    };
    vkCreateSampler(device, &tex_samp_info, nullptr, &texture_sampler);

    // Create shadow map
    createShadowMap();

    // Common vertex input
    VkVertexInputBindingDescription vb{.binding=0,.stride=sizeof(Vertex),.inputRate=VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[] = {
        {.location=0,.binding=0,.format=VK_FORMAT_R32G32B32_SFLOAT,.offset=offsetof(Vertex,position)},
        {.location=1,.binding=0,.format=VK_FORMAT_R32G32B32_SFLOAT,.offset=offsetof(Vertex,normal)},
        {.location=2,.binding=0,.format=VK_FORMAT_R32G32_SFLOAT,.offset=offsetof(Vertex,uv)},
    };
    VkPipelineVertexInputStateCreateInfo vi{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount=1,.pVertexBindingDescriptions=&vb,
        .vertexAttributeDescriptionCount=3,.pVertexAttributeDescriptions=attrs};
    VkPipelineInputAssemblyStateCreateInfo ia{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineRasterizationStateCreateInfo rast{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_BACK_BIT,
        .frontFace=VK_FRONT_FACE_CLOCKWISE,.lineWidth=1.0f};
    VkPipelineMultisampleStateCreateInfo ms{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba{
        .colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo cb{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount=1,.pAttachments=&cba};
    VkPipelineDepthStencilStateCreateInfo ds{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable=true,.depthWriteEnable=true,.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL};

    // Descriptor pool
    VkDescriptorPoolSize pools[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
    };
    VkDescriptorPoolCreateInfo dpool{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets=8,.poolSizeCount=4,.pPoolSizes=pools};
    vkCreateDescriptorPool(device, &dpool, nullptr, &descriptor_pool);

    // ===== Shadow pipeline =====
    {
        VkDescriptorSetLayoutBinding sb[] = {
            {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_VERTEX_BIT},
            {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_VERTEX_BIT},
        };
        VkDescriptorSetLayoutCreateInfo slc{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount=2,.pBindings=sb};
        vkCreateDescriptorSetLayout(device, &slc, nullptr, &shadow_descriptor_set_layout);

        VkDescriptorSetAllocateInfo sa{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&shadow_descriptor_set_layout};
        vkAllocateDescriptorSets(device, &sa, &shadow_descriptor_set);

        VkPipelineLayoutCreateInfo plc{.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount=1,.pSetLayouts=&shadow_descriptor_set_layout};
        vkCreatePipelineLayout(device, &plc, nullptr, &shadow_pipeline_layout);

        VkPipelineShaderStageCreateInfo stages[] = {
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=shadow_vert_module,.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=shadow_frag_module,.pName="main"},
        };

        VkViewport svp{.width=float(shadow_map_size),.height=float(shadow_map_size),.maxDepth=1};
        VkRect2D ssc{.extent={shadow_map_size,shadow_map_size}};
        VkPipelineViewportStateCreateInfo svs{.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount=1,.pViewports=&svp,.scissorCount=1,.pScissors=&ssc};

        VkPipelineRasterizationStateCreateInfo srast = rast;
        srast.depthBiasEnable = VK_TRUE;
        srast.depthBiasConstantFactor = 1.5f;
        srast.depthBiasSlopeFactor = 1.75f;

        VkPipelineColorBlendStateCreateInfo scb{.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};

        VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
        VkPipelineRenderingCreateInfoKHR dyn_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
            .depthAttachmentFormat = depth_format,
        };

        VkGraphicsPipelineCreateInfo spi{
            .sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &dyn_info,
            .stageCount=2,.pStages=stages,
            .pVertexInputState=&vi,.pInputAssemblyState=&ia,
            .pViewportState=&svs,.pRasterizationState=&srast,
            .pMultisampleState=&ms,.pDepthStencilState=&ds,
            .pColorBlendState=&scb,
            .layout=shadow_pipeline_layout,
            .renderPass=VK_NULL_HANDLE,
        };
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &spi, nullptr, &shadow_pipeline);
    }

    // ===== Main pipeline =====
    {
        VkDescriptorSetLayoutBinding mb[] = {
            {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding=2,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding=3,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT},
            {.binding=4,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.descriptorCount=1,
             .stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT},
        };
        VkDescriptorSetLayoutCreateInfo mlc{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount=5,.pBindings=mb};
        vkCreateDescriptorSetLayout(device, &mlc, nullptr, &descriptor_set_layout);

        VkDescriptorSetAllocateInfo ma{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&descriptor_set_layout};
        vkAllocateDescriptorSets(device, &ma, &descriptor_set);

        VkPipelineLayoutCreateInfo plc{.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount=1,.pSetLayouts=&descriptor_set_layout};
        vkCreatePipelineLayout(device, &plc, nullptr, &pipeline_layout);

        VkPipelineShaderStageCreateInfo stages[] = {
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vertex_shader_module,.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fragment_shader_module,.pName="main"},
        };
        VkViewport vp{.width=float(veekay::app.window_width),.height=float(veekay::app.window_height),.maxDepth=1};
        VkRect2D sc{.extent={veekay::app.window_width,veekay::app.window_height}};
        VkPipelineViewportStateCreateInfo vs{.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount=1,.pViewports=&vp,.scissorCount=1,.pScissors=&sc};

        VkGraphicsPipelineCreateInfo mpi{
            .sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount=2,.pStages=stages,
            .pVertexInputState=&vi,.pInputAssemblyState=&ia,
            .pViewportState=&vs,.pRasterizationState=&rast,
            .pMultisampleState=&ms,.pDepthStencilState=&ds,.pColorBlendState=&cb,
            .layout=pipeline_layout,.renderPass=veekay::app.vk_render_pass,
        };
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &mpi, nullptr, &pipeline);
    }

    // Buffers
    scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    model_uniforms_buffer = new veekay::graphics::Buffer(max_models*veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    spotlights_buffer = new veekay::graphics::Buffer(8*sizeof(Spotlight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    shadow_uniforms_buffer = new veekay::graphics::Buffer(sizeof(ShadowUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Write shadow descriptor set
    {
        VkDescriptorBufferInfo bi[] = {
            {.buffer=shadow_uniforms_buffer->buffer,.offset=0,.range=sizeof(ShadowUniforms)},
            {.buffer=model_uniforms_buffer->buffer,.offset=0,.range=sizeof(ModelUniforms)},
        };
        VkWriteDescriptorSet w[] = {
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=shadow_descriptor_set,
             .dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.pBufferInfo=&bi[0]},
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=shadow_descriptor_set,
             .dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.pBufferInfo=&bi[1]},
        };
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }

    // Write main descriptor set
    {
        VkDescriptorBufferInfo bi[] = {
            {.buffer=scene_uniforms_buffer->buffer,.offset=0,.range=sizeof(SceneUniforms)},
            {.buffer=model_uniforms_buffer->buffer,.offset=0,.range=sizeof(ModelUniforms)},
            {.buffer=spotlights_buffer->buffer,.offset=0,.range=VK_WHOLE_SIZE},
        };
        VkDescriptorImageInfo tex_img{.sampler=texture_sampler,.imageView=albedo_texture->view,
            .imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo shadow_img{.sampler=shadow_sampler,.imageView=shadow_image_view,
            .imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[] = {
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=descriptor_set,.dstBinding=0,.descriptorCount=1,
             .descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.pBufferInfo=&bi[0]},
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=descriptor_set,.dstBinding=1,.descriptorCount=1,
             .descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.pBufferInfo=&bi[1]},
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=descriptor_set,.dstBinding=2,.descriptorCount=1,
             .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[2]},
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=descriptor_set,.dstBinding=3,.descriptorCount=1,
             .descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&tex_img},
            {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=descriptor_set,.dstBinding=4,.descriptorCount=1,
             .descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&shadow_img},
        };
        vkUpdateDescriptorSets(device, 5, w, 0, nullptr);
    }

    // Meshes
    plane_mesh=generatePlane(); cube_mesh=generateCube();
    pyramid_mesh=generatePyramid(); sphere_mesh=generateSphere(20,20);

    // Scene
    models.push_back({plane_mesh,Transform{},{1,1,1},{.2f,.2f,.2f},4});
    models.push_back({cube_mesh,Transform{.position={-2.5f,-.5f,0}},{.5f,.7f,1},{1,1,1},32});
    models.push_back({pyramid_mesh,Transform{.position={0,-.3f,0},.scale={1.2f,1.2f,1.2f}},{1,.5f,.4f},{1,.8f,.8f},16});
    models.push_back({sphere_mesh,Transform{.position={2.5f,-.5f,0}},{.5f,1,.6f},{1,1,1},64});
    models.push_back({cube_mesh,Transform{.position={0,-.5f,-3},.scale={1.5f,1.5f,1.5f}},{1,1,1},{1,1,1},8});

    spotlights.push_back({.position={0,-4,2},.direction={0,1,-.3f},.color={1,.95f,.8f},.intensity=60,
        .inner_cutoff=std::cos(toRadians(15)),.outer_cutoff=std::cos(toRadians(25))});
    spotlights.push_back({.position={3,-3,3},.direction={-.5f,.7f,-.5f},.color={.3f,.5f,1},.intensity=40,
        .inner_cutoff=std::cos(toRadians(12)),.outer_cutoff=std::cos(toRadians(20))});
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;
    delete sphere_mesh.index_buffer; delete sphere_mesh.vertex_buffer;
    delete pyramid_mesh.index_buffer; delete pyramid_mesh.vertex_buffer;
    delete cube_mesh.index_buffer; delete cube_mesh.vertex_buffer;
    delete plane_mesh.index_buffer; delete plane_mesh.vertex_buffer;
    vkDestroySampler(device, texture_sampler, nullptr);
    delete albedo_texture;
    vkDestroySampler(device, shadow_sampler, nullptr);
    vkDestroyImageView(device, shadow_image_view, nullptr);
    vkFreeMemory(device, shadow_image_memory, nullptr);
    vkDestroyImage(device, shadow_image, nullptr);
    delete shadow_uniforms_buffer;
    delete spotlights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;
    vkDestroyDescriptorSetLayout(device, shadow_descriptor_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, shadow_pipeline, nullptr);
    vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, shadow_frag_module, nullptr);
    vkDestroyShaderModule(device, shadow_vert_module, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    (void)time;
    ImGui::Begin("Lighting Controls");
    ImGui::Text("WASD - move, Mouse LMB - look, Q/E - up/down");
    ImGui::Separator();

    static float ambient_intensity = 0.1f;
    ImGui::SliderFloat("Ambient", &ambient_intensity, 0, 1);

    static float dir_light_dir[3] = {0.3f, -1.0f, 0.2f};
    static float dir_light_color[3] = {1, 1, 0.9f};
    static float dir_light_intensity = 1.5f;
    ImGui::SliderFloat3("Light Dir", dir_light_dir, -1, 1);
    ImGui::ColorEdit3("Light Color", dir_light_color);
    ImGui::SliderFloat("Light Intensity", &dir_light_intensity, 0, 5);
    ImGui::Separator();

    ImGui::Text("Spotlights");
    static std::vector<std::pair<float,float>> sa;
    if(sa.size()!=spotlights.size()){sa.resize(spotlights.size());
        for(size_t i=0;i<spotlights.size();++i){
            sa[i].first=std::acos(clampVal(spotlights[i].inner_cutoff,0.f,1.f))*180.f/float(M_PI);
            sa[i].second=std::acos(clampVal(spotlights[i].outer_cutoff,0.f,1.f))*180.f/float(M_PI);}}

    for(size_t i=0;i<spotlights.size();++i){
        ImGui::PushID((int)i);
        if(ImGui::TreeNode("Spotlight","Spotlight %zu",i)){
            ImGui::SliderFloat3("Pos",&spotlights[i].position.x,-10,10);
            ImGui::SliderFloat3("Dir",&spotlights[i].direction.x,-1,1);
            spotlights[i].direction=veekay::vec3::normalized(spotlights[i].direction);
            ImGui::ColorEdit3("Col",&spotlights[i].color.x);
            ImGui::SliderFloat("Int",&spotlights[i].intensity,0,200);
            if(ImGui::SliderFloat("Inner",&sa[i].first,0,45)){
                spotlights[i].inner_cutoff=std::cos(toRadians(sa[i].first));
                if(sa[i].second<sa[i].first+1){sa[i].second=sa[i].first+1;
                    spotlights[i].outer_cutoff=std::cos(toRadians(sa[i].second));}}
            if(ImGui::SliderFloat("Outer",&sa[i].second,sa[i].first+1,45))
                spotlights[i].outer_cutoff=std::cos(toRadians(sa[i].second));
            if(ImGui::Button("Remove")){spotlights.erase(spotlights.begin()+i);sa.erase(sa.begin()+i);
                ImGui::TreePop();ImGui::PopID();ImGui::End();return;}
            ImGui::TreePop();}
        ImGui::PopID();}

    if(spotlights.size()<8&&ImGui::Button("Add Spotlight")){
        spotlights.push_back({.position={0,-3,0},.direction={0,1,0},.color={1,1,1},.intensity=50,
            .inner_cutoff=std::cos(toRadians(12.5f)),.outer_cutoff=std::cos(toRadians(17.5f))});
        sa.push_back({12.5f,17.5f});}

    ImGui::End();

    if(!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)){
        using namespace veekay::input;
        if(mouse::isButtonDown(mouse::Button::left)){
            auto d=mouse::cursorDelta();
            camera.rotation.y+=d.x*0.2f; camera.rotation.x-=d.y*0.2f;
            if(camera.rotation.x>89)camera.rotation.x=89;
            if(camera.rotation.x<-89)camera.rotation.x=-89;}
        auto v=camera.view();
        veekay::vec3 right={v.elements[0][0],v.elements[1][0],v.elements[2][0]};
        veekay::vec3 up={v.elements[0][1],v.elements[1][1],v.elements[2][1]};
        veekay::vec3 front={v.elements[0][2],v.elements[1][2],v.elements[2][2]};
        float sp=0.1f;
        if(keyboard::isKeyDown(keyboard::Key::w))camera.position+=front*sp;
        if(keyboard::isKeyDown(keyboard::Key::s))camera.position-=front*sp;
        if(keyboard::isKeyDown(keyboard::Key::d))camera.position+=right*sp;
        if(keyboard::isKeyDown(keyboard::Key::a))camera.position-=right*sp;
        if(keyboard::isKeyDown(keyboard::Key::q))camera.position-=up*sp;
        if(keyboard::isKeyDown(keyboard::Key::e))camera.position+=up*sp;}

    veekay::vec3 ld_vec = veekay::vec3::normalized({dir_light_dir[0],dir_light_dir[1],dir_light_dir[2]});
    veekay::mat4 light_vp = computeLightVP(ld_vec);

    float ar=float(veekay::app.window_width)/float(veekay::app.window_height);
    SceneUniforms su{
        .view_projection=camera.view_projection(ar),
        .light_view_projection=light_vp,
        .ambient_intensity=ambient_intensity,.point_light_count=0,
        .spotlight_count=(uint32_t)spotlights.size(),
        .light_direction=ld_vec,
        .light_color={dir_light_color[0],dir_light_color[1],dir_light_color[2]},
        .light_intensity=dir_light_intensity,
        .camera_position=camera.position};
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region=su;

    ShadowUniforms shu{.light_view_projection=light_vp};
    *(ShadowUniforms*)shadow_uniforms_buffer->mapped_region=shu;

    const size_t al=veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
    for(size_t i=0;i<models.size();++i){
        ModelUniforms mu; mu.model=models[i].transform.matrix();
        mu.albedo_color=models[i].albedo_color;
        mu.specular_color=models[i].specular_color;
        mu.shininess=models[i].shininess;
        *(ModelUniforms*)(static_cast<char*>(model_uniforms_buffer->mapped_region)+i*al)=mu;}

    if(!spotlights.empty())
        std::memcpy(spotlights_buffer->mapped_region,spotlights.data(),spotlights.size()*sizeof(Spotlight));
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo cbi{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd, &cbi);

    const size_t mal=veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
    VkDeviceSize zero=0;

    // ===== Shadow pass (Dynamic Rendering) =====
    {
        // Transition shadow image to depth attachment
        VkImageMemoryBarrier barrier{
            .sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask=0,.dstAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .image=shadow_image,
            .subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0,nullptr,0,nullptr,1,&barrier);

        VkRenderingAttachmentInfoKHR depth_att{
            .sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
            .imageView=shadow_image_view,
            .imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue={.depthStencil={1.0f,0}},
        };
        VkRenderingInfoKHR ri{
            .sType=VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
            .renderArea={.extent={shadow_map_size,shadow_map_size}},
            .layerCount=1,
            .pDepthAttachment=&depth_att,
        };
        pfnCmdBeginRendering(cmd, &ri);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
        for(size_t i=0;i<models.size();++i){
            const Mesh& m=models[i].mesh;
            vkCmdBindVertexBuffers(cmd,0,1,&m.vertex_buffer->buffer,&zero);
            vkCmdBindIndexBuffer(cmd,m.index_buffer->buffer,0,VK_INDEX_TYPE_UINT32);
            uint32_t off=(uint32_t)(i*mal);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,shadow_pipeline_layout,0,1,&shadow_descriptor_set,1,&off);
            vkCmdDrawIndexed(cmd,m.indices,1,0,0,0);}

        pfnCmdEndRendering(cmd);

        // Transition shadow image to shader read
        VkImageMemoryBarrier barrier2{
            .sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
            .oldLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .newLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .image=shadow_image,
            .subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,nullptr,0,nullptr,1,&barrier2);
    }

    // ===== Main pass =====
    {
        VkClearValue cc{.color={{.05f,.05f,.08f,1}}};
        VkClearValue cd{.depthStencil={1,0}};
        VkClearValue cv[]={cc,cd};
        VkRenderPassBeginInfo rp{.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass=veekay::app.vk_render_pass,.framebuffer=framebuffer,
            .renderArea={.extent={veekay::app.window_width,veekay::app.window_height}},
            .clearValueCount=2,.pClearValues=cv};
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        for(size_t i=0;i<models.size();++i){
            const Mesh& m=models[i].mesh;
            vkCmdBindVertexBuffers(cmd,0,1,&m.vertex_buffer->buffer,&zero);
            vkCmdBindIndexBuffer(cmd,m.index_buffer->buffer,0,VK_INDEX_TYPE_UINT32);
            uint32_t off=(uint32_t)(i*mal);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_layout,0,1,&descriptor_set,1,&off);
            vkCmdDrawIndexed(cmd,m.indices,1,0,0,0);}

        vkCmdEndRenderPass(cmd);
    }

    vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
    return veekay::run({
        .init=initialize, .shutdown=shutdown,
        .update=update, .render=render,
    });
}
