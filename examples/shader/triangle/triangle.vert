#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 offset;
layout (location = 3) in vec4 bla;


layout (location = 0) out vec4 o_color;
void main() {
    o_color = color;
    float x =  (bla.x * pos.x) + offset.x;
    float y =  (bla.y * pos.y) + offset.y;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
