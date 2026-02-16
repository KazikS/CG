#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {
	constexpr float camera_fov = 70.0f;
	constexpr float camera_near_plane = 0.01f;
	constexpr float camera_far_plane = 100.0f;

	struct Matrix {
		float m[4][4];
	};

	struct Vector {
		float x, y, z;
	};

	struct Vertex {
		Vector position;
		Vector color;
	};

	struct ShaderConstants {
		Matrix projection;
		Matrix transform;
		Vector color;
	};

	struct VulkanBuffer {
		VkBuffer buffer;
		VkDeviceMemory memory;
	};

	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	VulkanBuffer cube_vb{}, cube_ib{};
	uint32_t cube_index_count = 0;

	VulkanBuffer pyramid_vb{}, pyramid_ib{};
	uint32_t pyramid_index_count = 0;

	VulkanBuffer sphere_vb{}, sphere_ib{};
	uint32_t sphere_index_count = 0;

	Vector cube_position    = { -2.5f, 0.0f, 5.0f };
	Vector cube_color       = { 0.2f, 0.6f, 1.0f };

	Vector pyramid_position = { 0.0f, 0.0f, 5.0f };
	Vector pyramid_color    = { 1.0f, 0.3f, 0.2f };

	Vector sphere_position  = { 2.5f, 0.0f, 5.0f };
	Vector sphere_color     = { 0.2f, 1.0f, 0.4f };

	// ========== Matrix helpers ==========

	Matrix zeroMatrix() {
		Matrix r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				r.m[i][j] = 0.0f;
		return r;
	}

	Matrix identity() {
		Matrix result = zeroMatrix();
		result.m[0][0] = 1.0f;
		result.m[1][1] = 1.0f;
		result.m[2][2] = 1.0f;
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix projectionMatrix(float fov, float aspect_ratio, float near_plane, float far_plane) {
		Matrix result = zeroMatrix();
		const float radians = fov * static_cast<float>(M_PI) / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);
		result.m[0][0] = cot / aspect_ratio;
		result.m[1][1] = cot;
		result.m[2][3] = 1.0f;
		result.m[2][2] = far_plane / (far_plane - near_plane);
		result.m[3][2] = (-near_plane * far_plane) / (far_plane - near_plane);
		return result;
	}

	Matrix translation(Vector v) {
		Matrix result = identity();
		result.m[3][0] = v.x;
		result.m[3][1] = v.y;
		result.m[3][2] = v.z;
		return result;
	}

	Matrix rotation(Vector axis, float angle) {
		Matrix result = zeroMatrix();
		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
		if (length == 0.0f) return identity();
		axis.x /= length; axis.y /= length; axis.z /= length;
		float sina = sinf(angle), cosa = cosf(angle), cosv = 1.0f - cosa;
		result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
		result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);
		result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
		result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);
		result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result.m[2][2] = (axis.z * axis.z * cosv) + cosa;
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix multiply(const Matrix& a, const Matrix& b) {
		Matrix result = zeroMatrix();
		for (int j = 0; j < 4; j++)
			for (int i = 0; i < 4; i++)
				for (int k = 0; k < 4; k++)
					result.m[j][i] += a.m[j][k] * b.m[k][i];
		return result;
	}

	// ========== Shader loading ==========

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

	// ========== Buffer creation ==========

	VulkanBuffer createBuffer(size_t size, void* data, VkBufferUsageFlags usage) {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
		VulkanBuffer result{};

		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n"; return {};
		}

		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			if ((requirements.memoryTypeBits & (1u << i)) &&
				(properties.memoryTypes[i].propertyFlags & flags) == flags) {
				index = i; break;
			}
		}
		if (index == UINT_MAX) { std::cerr << "No suitable memory type\n"; return {}; }

		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};
		if (vkAllocateMemory(device, &allocInfo, nullptr, &result.memory) != VK_SUCCESS) return {};
		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) return {};

		void* device_data = nullptr;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);
		if (device_data && data && size > 0) memcpy(device_data, data, size);
		vkUnmapMemory(device, result.memory);

		return result;
	}

	void destroyBuffer(const VulkanBuffer& buffer) {
		VkDevice& device = veekay::app.vk_device;
		if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
		if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
	}

	// ========== Geometry generation ==========

	void generateCube(std::vector<Vertex>& verts, std::vector<uint32_t>& inds) {
		Vector positions[8] = {
			{-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
			{ 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
			{-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
			{ 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
		};
		Vector colors[8] = {
			{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f},
			{0.2f, 0.4f, 1.0f}, {0.0f, 0.3f, 0.8f},
			{0.1f, 0.2f, 0.7f}, {0.3f, 0.5f, 1.0f},
			{0.5f, 0.7f, 1.0f}, {0.1f, 0.4f, 0.9f},
		};
		verts.clear(); inds.clear();
		for (int i = 0; i < 8; ++i)
			verts.push_back({ positions[i], colors[i] });
		uint32_t cubeIndices[] = {
			0, 1, 2,  0, 2, 3,
			5, 4, 7,  5, 7, 6,
			4, 0, 3,  4, 3, 7,
			1, 5, 6,  1, 6, 2,
			3, 2, 6,  3, 6, 7,
			4, 5, 1,  4, 1, 0,
		};
		inds.assign(cubeIndices, cubeIndices + 36);
	}

	void generatePyramid(std::vector<Vertex>& verts, std::vector<uint32_t>& inds) {
		Vector positions[5] = {
			{-0.5f, -0.5f, -0.5f},
			{ 0.5f, -0.5f, -0.5f},
			{ 0.5f, -0.5f,  0.5f},
			{-0.5f, -0.5f,  0.5f},
			{ 0.0f,  0.5f,  0.0f},
		};
		Vector colors[5] = {
			{1.0f, 0.2f, 0.1f},
			{1.0f, 0.4f, 0.1f},
			{0.8f, 0.2f, 0.0f},
			{1.0f, 0.5f, 0.2f},
			{1.0f, 1.0f, 0.8f},
		};
		verts.clear(); inds.clear();
		for (int i = 0; i < 5; ++i)
			verts.push_back({ positions[i], colors[i] });
		uint32_t pyramidIndices[] = {
			0, 1, 2,  0, 2, 3,
			0, 1, 4,
			1, 2, 4,
			2, 3, 4,
			3, 0, 4,
		};
		inds.assign(pyramidIndices, pyramidIndices + 18);
	}

	void generateSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& inds) {
		const int stacks = 20;
		const int slices = 20;
		const float radius = 0.5f;
		verts.clear(); inds.clear();
		for (int i = 0; i <= stacks; ++i) {
			float v = float(i) / float(stacks);
			float phi = v * static_cast<float>(M_PI);
			float sinPhi = sinf(phi), cosPhi = cosf(phi);
			for (int j = 0; j <= slices; ++j) {
				float u = float(j) / float(slices);
				float theta = u * 2.0f * static_cast<float>(M_PI);
				float x = sinPhi * cosf(theta) * radius;
				float y = cosPhi * radius;
				float z = sinPhi * sinf(theta) * radius;
				Vertex vert;
				vert.position = { x, y, z };
				vert.color = {
					(x / radius + 1.0f) * 0.5f * 0.2f,
					(y / radius + 1.0f) * 0.5f * 1.0f,
					(z / radius + 1.0f) * 0.5f * 0.4f
				};
				verts.push_back(vert);
			}
		}
		for (int i = 0; i < stacks; ++i) {
			for (int j = 0; j < slices; ++j) {
				uint32_t first = uint32_t(i * (slices + 1) + j);
				uint32_t second = first + uint32_t(slices + 1);
				inds.push_back(first);     inds.push_back(second);     inds.push_back(first + 1);
				inds.push_back(second);    inds.push_back(second + 1); inds.push_back(first + 1);
			}
		}
	}

	void uploadGeometry(const std::vector<Vertex>& verts, const std::vector<uint32_t>& inds,
						VulkanBuffer& vb, VulkanBuffer& ib, uint32_t& indexCount) {
		if (!verts.empty())
			vb = createBuffer(verts.size() * sizeof(Vertex),
				const_cast<void*>(static_cast<const void*>(verts.data())),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		if (!inds.empty())
			ib = createBuffer(inds.size() * sizeof(uint32_t),
				const_cast<void*>(static_cast<const void*>(inds.data())),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		indexCount = static_cast<uint32_t>(inds.size());
	}

	// ========== Draw a single object ==========

	void drawObject(VkCommandBuffer cmd, const Matrix& proj,
					VulkanBuffer& vb, VulkanBuffer& ib, uint32_t indexCount,
					Vector position, Vector color) {
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &offset);
		vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

		Matrix transform = translation(position);

		ShaderConstants constants{
			.projection = proj,
			.transform = transform,
			.color = color,
		};

		vkCmdPushConstants(cmd, pipeline_layout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(ShaderConstants), &constants);

		vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
	}

	// ========== Application callbacks ==========

	void initialize() {
		VkDevice& device = veekay::app.vk_device;

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

		{
			VkPipelineShaderStageCreateInfo stage_infos[2];
			stage_infos[0] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertex_shader_module,
				.pName = "main",
			};
			stage_infos[1] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragment_shader_module,
				.pName = "main",
			};

			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};
			VkVertexInputAttributeDescription attributes[] = {
				{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
				{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
			};
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = 2,
				.pVertexAttributeDescriptions = attributes,
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
				.sampleShadingEnable = false,
				.minSampleShading = 1.0f,
			};
			VkViewport viewport{
				.x = 0.0f, .y = 0.0f,
				.width = static_cast<float>(veekay::app.window_width),
				.height = static_cast<float>(veekay::app.window_height),
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};
			VkRect2D scissor{
				.offset = {0, 0},
				.extent = {veekay::app.window_width, veekay::app.window_height},
			};
			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1, .pViewports = &viewport,
				.scissorCount = 1, .pScissors = &scissor,
			};
			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = true,
				.depthWriteEnable = true,
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
			VkPushConstantRange push_constants{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				.size = sizeof(ShaderConstants),
			};
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_constants,
			};
			if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create pipeline layout\n";
				veekay::app.running = false; return;
			}
			VkGraphicsPipelineCreateInfo pipelineInfo{
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
			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
				std::cerr << "Failed to create graphics pipeline\n";
				veekay::app.running = false; return;
			}
		}

		{
			std::vector<Vertex> verts;
			std::vector<uint32_t> inds;

			generateCube(verts, inds);
			uploadGeometry(verts, inds, cube_vb, cube_ib, cube_index_count);

			generatePyramid(verts, inds);
			uploadGeometry(verts, inds, pyramid_vb, pyramid_ib, pyramid_index_count);

			generateSphere(verts, inds);
			uploadGeometry(verts, inds, sphere_vb, sphere_ib, sphere_index_count);
		}
	}

	void shutdown() {
		VkDevice& device = veekay::app.vk_device;
		destroyBuffer(cube_ib);    destroyBuffer(cube_vb);
		destroyBuffer(pyramid_ib); destroyBuffer(pyramid_vb);
		destroyBuffer(sphere_ib);  destroyBuffer(sphere_vb);
		if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
		if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		if (fragment_shader_module) vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		if (vertex_shader_module) vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time) {
		(void)time;

		ImGui::Begin("Scene Controls");

		if (ImGui::CollapsingHeader("Cube", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::PushID("cube");
			ImGui::DragFloat3("Position", reinterpret_cast<float*>(&cube_position), 0.05f);
			ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&cube_color));
			ImGui::PopID();
		}

		if (ImGui::CollapsingHeader("Pyramid", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::PushID("pyramid");
			ImGui::DragFloat3("Position", reinterpret_cast<float*>(&pyramid_position), 0.05f);
			ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&pyramid_color));
			ImGui::PopID();
		}

		if (ImGui::CollapsingHeader("Sphere", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::PushID("sphere");
			ImGui::DragFloat3("Position", reinterpret_cast<float*>(&sphere_position), 0.05f);
			ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&sphere_color));
			ImGui::PopID();
		}

		ImGui::End();
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
			VkClearValue clear_color{ .color = {{0.08f, 0.08f, 0.12f, 1.0f}} };
			VkClearValue clear_depth{ .depthStencil = {1.0f, 0} };
			VkClearValue clear_values[] = { clear_color, clear_depth };
			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = { .extent = { veekay::app.window_width, veekay::app.window_height } },
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};
			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		Matrix proj = projectionMatrix(camera_fov,
			float(veekay::app.window_width) / float(veekay::app.window_height),
			camera_near_plane, camera_far_plane);

		drawObject(cmd, proj, cube_vb,    cube_ib,    cube_index_count,    cube_position,    cube_color);
		drawObject(cmd, proj, pyramid_vb, pyramid_ib, pyramid_index_count, pyramid_position, pyramid_color);
		drawObject(cmd, proj, sphere_vb,  sphere_ib,  sphere_index_count,  sphere_position,  sphere_color);

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
