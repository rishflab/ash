use ash::util::*;
use ash::vk;
use examples::*;
use std::default::Default;
use std::ffi::CStr;
use std::io::Cursor;
use std::mem;
use std::mem::align_of;

const INSTANCE_COUNT_X: u64 = 16;
const INSTANCE_COUNT_Y: u64 = 9;
const VERTEX_COUNT: u64 = 500;

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

fn main() {
    unsafe {
        let base = ExampleBase::new(1280, 720);
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: base.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpass = vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let renderpass = base
            .device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap();

        let framebuffers: Vec<vk::Framebuffer> = base
            .present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, base.depth_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderpass)
                    .attachments(&framebuffer_attachments)
                    .width(base.surface_resolution.width)
                    .height(base.surface_resolution.height)
                    .layers(1);

                base.device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            })
            .collect();
        
        let mut instance_buffer_data = vec![];
        for j in 0..INSTANCE_COUNT_Y {
            for i in 0..INSTANCE_COUNT_X {
                let x_incr = 1.0/INSTANCE_COUNT_X as f32;
                let y_incr = 1.0/INSTANCE_COUNT_Y as f32;
                let x_offset = -1.0 + x_incr + 2.0 * x_incr * i as f32;
                let y_offset = -1.0 + y_incr + 2.0 * y_incr * j as f32;
                instance_buffer_data.push(Vertex {
                    pos: [x_offset, y_offset, 0.0, 1.0],
                    color: [x_incr, y_incr, 0.0, 1.0],
                });
            }
        }

        let instance_buffer_info = vk::BufferCreateInfo::builder()
            .size(INSTANCE_COUNT_X * INSTANCE_COUNT_Y * std::mem::size_of::<Vertex>() as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let instance_input_buffer = base
            .device
            .create_buffer(&instance_buffer_info, None)
            .unwrap();

        let instance_input_buffer_memory_req = base
            .device
            .get_buffer_memory_requirements(instance_input_buffer);

        let instance_input_buffer_memory_index = find_memorytype_index(
            &instance_input_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the instance buffer.");

        let instance_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: instance_input_buffer_memory_req.size,
            memory_type_index: instance_input_buffer_memory_index,
            ..Default::default()
        };

        let instance_input_buffer_memory = base
            .device
            .allocate_memory(&instance_buffer_allocate_info, None)
            .unwrap();

        let instance_ptr = base
            .device
            .map_memory(
                instance_input_buffer_memory,
                0,
                instance_input_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        let mut instance_align = Align::new(
            instance_ptr,
            align_of::<Vertex>() as u64,
            instance_input_buffer_memory_req.size,
        );
        instance_align.copy_from_slice(&instance_buffer_data);
        base.device.unmap_memory(instance_input_buffer_memory);
        base.device
            .bind_buffer_memory(instance_input_buffer, instance_input_buffer_memory, 0)
            .unwrap();

        let edge = 1.2/VERTEX_COUNT as f32; 
        let (vertices, index_buffer_data) = gen_plane(edge, VERTEX_COUNT as u32, VERTEX_COUNT as u32);
        let index_buffer_info = vk::BufferCreateInfo::builder()
            .size((std::mem::size_of::<u32>() * index_buffer_data.len()) as u64)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let index_buffer = base.device.create_buffer(&index_buffer_info, None).unwrap();
        let index_buffer_memory_req = base.device.get_buffer_memory_requirements(index_buffer);
        let index_buffer_memory_index = find_memorytype_index(
            &index_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the index buffer.");

        let index_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: index_buffer_memory_req.size,
            memory_type_index: index_buffer_memory_index,
            ..Default::default()
        };
        let index_buffer_memory = base
            .device
            .allocate_memory(&index_allocate_info, None)
            .unwrap();
        let index_ptr = base
            .device
            .map_memory(
                index_buffer_memory,
                0,
                index_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut index_slice = Align::new(
            index_ptr,
            align_of::<u32>() as u64,
            index_buffer_memory_req.size,
        );
        index_slice.copy_from_slice(&index_buffer_data);
        base.device.unmap_memory(index_buffer_memory);
        base.device
            .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
            .unwrap();

        let vertex_input_buffer_info = vk::BufferCreateInfo {
            size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vertex_input_buffer = base
            .device
            .create_buffer(&vertex_input_buffer_info, None)
            .unwrap();

        let vertex_input_buffer_memory_req = base
            .device
            .get_buffer_memory_requirements(vertex_input_buffer);

        let vertex_input_buffer_memory_index = find_memorytype_index(
            &vertex_input_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the vertex buffer.");

        let vertex_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: vertex_input_buffer_memory_req.size,
            memory_type_index: vertex_input_buffer_memory_index,
            ..Default::default()
        };

        let vertex_input_buffer_memory = base
            .device
            .allocate_memory(&vertex_buffer_allocate_info, None)
            .unwrap();

        let vert_ptr = base
            .device
            .map_memory(
                vertex_input_buffer_memory,
                0,
                vertex_input_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        let mut vert_align = Align::new(
            vert_ptr,
            align_of::<Vertex>() as u64,
            vertex_input_buffer_memory_req.size,
        );
        vert_align.copy_from_slice(&vertices);
        base.device.unmap_memory(vertex_input_buffer_memory);
        base.device
            .bind_buffer_memory(vertex_input_buffer, vertex_input_buffer_memory, 0)
            .unwrap();
        let mut vertex_spv_file = Cursor::new(&include_bytes!("../../shader/triangle/vert.spv")[..]);
        let mut frag_spv_file = Cursor::new(&include_bytes!("../../shader/triangle/frag.spv")[..]);

        let vertex_code =
            read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

        let frag_code =
            read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

        let vertex_shader_module = base
            .device
            .create_shader_module(&vertex_shader_info, None)
            .expect("Vertex shader module error");

        let fragment_shader_module = base
            .device
            .create_shader_module(&frag_shader_info, None)
            .expect("Fragment shader module error");

        let layout_create_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = base
            .device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let vertex_input_binding_descriptions = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::INSTANCE,
            },
        ];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_input_binding_descriptions);
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: base.surface_resolution.width as f32,
            height: base.surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [base.surface_resolution.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(&scissors)
            .viewports(&viewports);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(renderpass);

        let graphics_pipelines = base
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_info.build()],
                None,
            )
            .expect("Unable to create graphics pipeline");

        let graphic_pipeline = graphics_pipelines[0];

        base.render_loop(|| {
            let (present_index, _) = base
                .swapchain_loader
                .acquire_next_image(
                    base.swapchain,
                    std::u64::MAX,
                    base.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(renderpass)
                .framebuffer(framebuffers[present_index as usize])
                .render_area(base.surface_resolution.into())
                .clear_values(&clear_values);

            record_submit_commandbuffer(
                &base.device,
                base.draw_command_buffer,
                base.draw_commands_reuse_fence,
                base.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[base.present_complete_semaphore],
                &[base.rendering_complete_semaphore],
                |device, draw_command_buffer| {
                    device.cmd_begin_render_pass(
                        draw_command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    device.cmd_bind_pipeline(
                        draw_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphic_pipeline,
                    );
                    device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                    device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                    device.cmd_bind_vertex_buffers(
                        draw_command_buffer,
                        0,
                        &[vertex_input_buffer],
                        &[0],
                    );
                    device.cmd_bind_vertex_buffers(
                        draw_command_buffer,
                        1,
                        &[instance_input_buffer],
                        &[0],
                    );
                    device.cmd_bind_index_buffer(
                        draw_command_buffer,
                        index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(
                        draw_command_buffer,
                        index_buffer_data.len() as u32,
                        INSTANCE_COUNT_X as u32 * INSTANCE_COUNT_Y as u32,
                        0,
                        0,
                        0,
                    );
                    // Or draw without the index buffer
                    // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
                    device.cmd_end_render_pass(draw_command_buffer);
                },
            );
            //let mut present_info_err = mem::zeroed();
            let wait_semaphors = [base.rendering_complete_semaphore];
            let swapchains = [base.swapchain];
            let image_indices = [present_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            base.swapchain_loader
                .queue_present(base.present_queue, &present_info)
                .unwrap();
        });

        base.device.device_wait_idle().unwrap();
        for pipeline in graphics_pipelines {
            base.device.destroy_pipeline(pipeline, None);
        }
        base.device.destroy_pipeline_layout(pipeline_layout, None);
        base.device
            .destroy_shader_module(vertex_shader_module, None);
        base.device
            .destroy_shader_module(fragment_shader_module, None);
        base.device.free_memory(index_buffer_memory, None);
        base.device.destroy_buffer(index_buffer, None);
        base.device.free_memory(vertex_input_buffer_memory, None);
        base.device.destroy_buffer(vertex_input_buffer, None);
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(renderpass, None);
    }
}



fn gen_plane(
    triangle_edge: f32,
    w: u32,
    h: u32,
) -> (Vec<Vertex>, Vec<u32>) {
    use std::ops::Neg;

    let mut vertices = vec![];

    let start_x = triangle_edge.neg() * w as f32 / 2.0;
    let start_y = triangle_edge.neg() * h as f32 / 2.0;

    for j in 0..h + 1 {
        for i in 0..w + 1 {
            let x = start_x + i as f32 * triangle_edge;
            let y = start_y + j as f32 * triangle_edge;
            vertices.push(Vertex {
                pos: [x, y, 0.0, 1.0],
                color: [x , y, (x+y)/2.0, 1.0],
            });
        }
    }

    let mut indices = vec![];

    for j in 0..h {
        for i in 0..w {
            let row = j * (w + 1);
            let next_row = (j + 1) * (w + 1);
            indices.push(row + i);
            indices.push(row + i + 1);
            indices.push(next_row + i + 1);
            indices.push(row + i);
            indices.push(next_row + i + 1);
            indices.push(next_row + i);
        }
    }

    round_flat_plane_vertices(&mut vertices);

    (vertices, indices)
}


fn round_flat_plane_vertices(vertices: &mut Vec<Vertex>) {
    for Vertex {
        pos: p,
        ..
    } in vertices.iter_mut()
    {
        //round the vertices to 2 decimal places to reduce leaks/overlap in mesh
        p[0] = (p[0] * 100.0).round() / 100.0;
        p[1] = (p[1] * 100.0).round() / 100.0;
        p[2] = (p[2] * 100.0).round() / 100.0;
    }
}