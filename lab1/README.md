# Компьютерная графика — Лабораторная 1 (Вариант 10)

3D-сцена с кубом, пирамидой и сферой. Vulkan + GLFW + ImGui.

## Требования

- Windows 10/11
- Visual Studio 2022 (компонент C++ Desktop)
- CMake >= 3.20
- Ninja
- Vulkan SDK (https://vulkan.lunarg.com)

## Сборка и запуск

Открыть **x64 Native Tools Command Prompt for VS 2022**, затем:
```bash
cmake --preset debug
cmake --build build-debug --parallel
.\build-debug\testbed\testbed.exe
```