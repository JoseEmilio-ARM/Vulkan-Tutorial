#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>

#if 1 //def __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#ifndef LAB_TASK_LEVEL
#define LAB_TASK_LEVEL 1
#endif

#define LAB_TASK_AS_BUILD_AND_BIND 4
#define LAB_TASK_AS_ANIMATION 6
#define LAB_TASK_AS_OPAQUE_FLAG 7
#define LAB_TASK_INSTANCE_LUT 9
#define LAB_TASK_REFLECTIONS 11

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
const std::string MODEL_PATH = "models/plant_on_table.obj";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
            vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) ),
            vk::VertexInputAttributeDescription( 3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) )
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
    }
};

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        auto h = std::hash<glm::vec3>()(vertex.pos) ^ (std::hash<glm::vec3>()(vertex.color) << 1);
        h = (h >> 1) ^ (std::hash<glm::vec2>()(vertex.texCoord) << 1);
        h = (h >> 1) ^ (std::hash<glm::vec3>()(vertex.normal) << 1);
        return h;
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPos;
};

struct PushConstant {
    uint32_t materialIndex;
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
    // TASK11
    uint32_t reflective;
#endif // LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context  context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    std::vector<vk::raii::Image> textureImages;
    std::vector<vk::raii::DeviceMemory> textureImageMemories;
    std::vector<vk::raii::ImageView> textureImageViews;
    vk::raii::Sampler textureSampler = nullptr;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    vk::raii::Buffer uvBuffer = nullptr;
    vk::raii::DeviceMemory uvBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> blasBuffers;
    std::vector<vk::raii::DeviceMemory> blasMemories;
    std::vector<vk::raii::AccelerationStructureKHR> blasHandles;

    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    vk::raii::Buffer instanceBuffer = nullptr;
    vk::raii::DeviceMemory instanceMemory = nullptr;

    vk::raii::Buffer tlasBuffer = nullptr;
    vk::raii::DeviceMemory tlasMemory = nullptr;
    vk::raii::Buffer tlasScratchBuffer = nullptr;
    vk::raii::DeviceMemory tlasScratchMemory = nullptr;
    vk::raii::AccelerationStructureKHR tlas = nullptr;

    struct InstanceLUT {
        uint32_t materialID;
        uint32_t indexBufferOffset;
    };
    std::vector<InstanceLUT> instanceLUTs;
    vk::raii::Buffer instanceLUTBuffer = nullptr;
    vk::raii::DeviceMemory instanceLUTBufferMemory = nullptr;

    UniformBufferObject ubo{};

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    struct SubMesh {
        uint32_t indexOffset;
        uint32_t indexCount;
        int materialID;
        uint32_t firstVertex;
        uint32_t maxVertex;
        bool alphaCut;
        bool reflective;
    };
    std::vector<SubMesh> submeshes;
    std::vector<tinyobj::material_t> materials;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> globalDescriptorSets;
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t semaphoreIndex = 0;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName,
        vk::KHRAccelerationStructureExtensionName,
        vk::KHRBufferDeviceAddressExtensionName,
        vk::KHRDeferredHostOperationsExtensionName,
        vk::KHRRayQueryExtensionName
    };

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createCommandPool();
        loadModel();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUVBuffer();
        createAccelerationStructures();
        createInstanceLUTBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void cleanup() const {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
    }

    void createInstance() {
        /*constexpr */vk::ApplicationInfo appInfo = { };
        appInfo .pApplicationName   = "Hello Triangle",
                    appInfo.applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                    appInfo.pEngineName        = "No Engine",
                    appInfo.engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                    appInfo.apiVersion         = vk::ApiVersion14;

        // Get the required layers
        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) {
          requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        for (auto const& requiredLayer : requiredLayers)
        {
            if (std::ranges::none_of(layerProperties,
                                     [requiredLayer](auto const& layerProperty)
                                     { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
            {
                throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
            }
        }

        // Get the required extensions.
        auto requiredExtensions = getRequiredExtensions();

        // Check if the required extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const& requiredExtension : requiredExtensions)
        {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const& extensionProperty)
                                     { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }

        vk::InstanceCreateInfo createInfo = { };
            createInfo.pApplicationInfo        = &appInfo,
            createInfo.enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            createInfo.ppEnabledLayerNames     = requiredLayers.data(),
            createInfo.enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        instance = vk::raii::Instance(context, createInfo);
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT;
            debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags,
            debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags,
            debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;

        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto                            devIter = std::ranges::find_if(
          devices,
          [&]( auto const & device )
          {
            // Check if the device supports the Vulkan 1.3 API version
            bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

            // Check if any of the queue families support graphics operations
            auto queueFamilies = device.getQueueFamilyProperties();
            bool supportsGraphics =
              std::ranges::any_of( queueFamilies, []( auto const & qfp ) { return !!( qfp.queueFlags & vk::QueueFlagBits::eGraphics ); } );

            // Check if all required device extensions are available
            auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
            bool supportsAllRequiredExtensions =
              std::ranges::all_of( requiredDeviceExtension,
                                   [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
                                   {
                                     return std::ranges::any_of( availableDeviceExtensions,
                                                                 [requiredDeviceExtension]( auto const & availableDeviceExtension )
                                                                 { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; } );
                                   } );

            auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan12Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                vk::PhysicalDeviceRayQueryFeaturesKHR>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().shaderSampledImageArrayNonUniformIndexing &&
                                            features.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress &&
                                            features.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure &&
                                            features.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;

            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
          } );
        if ( devIter != devices.end() )
        {
            physicalDevice = *devIter;
        }
        else
        {
            throw std::runtime_error( "failed to find a suitable GPU!" );
        }
    }

    void createLogicalDevice() {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::ranges::find_if( queueFamilyProperties, []( auto const & qfp )
                        { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); } );

        graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        auto presentIndex = physicalDevice.getSurfaceSupportKHR( graphicsIndex, *surface )
                                           ? graphicsIndex
                                           : ~0;
        if ( presentIndex == queueFamilyProperties.size() )
        {
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
            {
                if ( ( queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
                     physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                {
                    graphicsIndex = static_cast<uint32_t>( i );
                    presentIndex  = graphicsIndex;
                    break;
                }
            }
            if ( presentIndex == queueFamilyProperties.size() )
            {
                // there's nothing like a single family index that supports both graphics and present -> look for another
                // family index that supports present
                for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
                {
                    if ( physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                    {
                        presentIndex = static_cast<uint32_t>( i );
                        break;
                    }
                }
            }
        }
        if ( ( graphicsIndex == queueFamilyProperties.size() ) || ( presentIndex == queueFamilyProperties.size() ) )
        {
            throw std::runtime_error( "Could not find a queue for graphics or present -> terminating" );
        }

        vk::PhysicalDeviceFeatures2 f2 = { };
                f2.features.samplerAnisotropy = true;                       // vk::PhysicalDeviceFeatures2
        vk::PhysicalDeviceVulkan12Features f12 = { };
         f12.shaderSampledImageArrayNonUniformIndexing = true, f12.descriptorBindingSampledImageUpdateAfterBind = true,
                 f12.descriptorBindingPartiallyBound = true, f12.descriptorBindingVariableDescriptorCount = true,
                 f12.runtimeDescriptorArray = true, f12.bufferDeviceAddress = true; // },    // vk::PhysicalDeviceVulkan12Features
        vk::PhysicalDeviceVulkan13Features f13 = { };
                f13.synchronization2 = true, f13.dynamicRendering = true; //,             // vk::PhysicalDeviceVulkan13Features
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT dS = { };
                dS.extendedDynamicState = true;                                   // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR as = { };
                as.accelerationStructure = true;                                  // vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        vk::PhysicalDeviceRayQueryFeaturesKHR rq = { };
                rq.rayQuery = true;                                                // vk::PhysicalDeviceRayQueryFeaturesKHR

        // query for Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR> featureChain = {
               f2, f12, f13, dS, as, rq
        };

        // create a Device
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo { };
        deviceQueueCreateInfo.queueFamilyIndex = graphicsIndex,
        deviceQueueCreateInfo.queueCount = 1, deviceQueueCreateInfo.pQueuePriorities = &queuePriority;
        vk::DeviceCreateInfo      deviceCreateInfo = { }; deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                                    deviceCreateInfo.queueCreateInfoCount = 1,
                                                    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo,
                                                    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                                                    deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

        device = vk::raii::Device( physicalDevice, deviceCreateInfo );
        graphicsQueue = vk::raii::Queue( device, graphicsIndex, 0 );
        presentQueue = vk::raii::Queue( device, presentIndex, 0 );
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR( surface ));
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo = { };
            swapChainCreateInfo.surface = surface, swapChainCreateInfo.minImageCount = minImageCount,
            swapChainCreateInfo.imageFormat = swapChainImageFormat, swapChainCreateInfo.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
            swapChainCreateInfo.imageExtent = swapChainExtent, swapChainCreateInfo.imageArrayLayers =1,
            swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment, swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive,
            swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform, swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            swapChainCreateInfo.presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
            swapChainCreateInfo.clipped = true;

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo { };
            imageViewCreateInfo.viewType = vk::ImageViewType::e2D,
            imageViewCreateInfo.format = swapChainImageFormat,
            imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        ;
        for ( auto image : swapChainImages )
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back( device, imageViewCreateInfo );
        }
    }

    void createDescriptorSetLayout() {
        // Use descriptor set 0 for global data
        // TASK04: The acceleration structure uses binding 1
        std::array global_bindings = {
            vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding( 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding( 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding( 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        vk::DescriptorSetLayoutCreateInfo globalLayoutInfo = { };
         globalLayoutInfo.bindingCount = static_cast<uint32_t>(global_bindings.size()), globalLayoutInfo.pBindings = global_bindings.data();

        descriptorSetLayoutGlobal = vk::raii::DescriptorSetLayout(device, globalLayoutInfo);

        // Use descriptor set 1 for bindless material data
        uint32_t textureCount = static_cast<uint32_t>(textureImageViews.size());

        std::array material_bindings = {
            vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eSampledImage, static_cast<uint32_t>(textureCount), vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        std::vector<vk::DescriptorBindingFlags> bindingFlags = {
            vk::DescriptorBindingFlagBits::eUpdateAfterBind,
            vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::eUpdateAfterBind
        };

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = { };
            flagsCreateInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
            flagsCreateInfo.pBindingFlags = bindingFlags.data()
        ;

        vk::DescriptorSetLayoutCreateInfo materialLayoutInfo = { };
            materialLayoutInfo.pNext = &flagsCreateInfo,
            materialLayoutInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
            materialLayoutInfo.bindingCount = static_cast<uint32_t>(material_bindings.size()),
            materialLayoutInfo.pBindings = material_bindings.data()
        ;

        descriptorSetLayoutMaterial = vk::raii::DescriptorSetLayout(device, materialLayoutInfo);
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo = { };
        vertShaderStageInfo .stage = vk::ShaderStageFlagBits::eVertex, vertShaderStageInfo.module = shaderModule,  vertShaderStageInfo.pName = "vertMain";
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo = { };
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment, fragShaderStageInfo.module = shaderModule, fragShaderStageInfo.pName = "fragMain";
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.vertexBindingDescriptionCount = 1,
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription,
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data()
        ;
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = { };
            inputAssembly.topology = vk::PrimitiveTopology::eTriangleList,
            inputAssembly.primitiveRestartEnable = vk::False
        ;
        vk::PipelineViewportStateCreateInfo viewportState = { };
            viewportState.viewportCount = 1,
            viewportState.scissorCount = 1
        ;
        vk::PipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.depthClampEnable = vk::False,
            rasterizer.rasterizerDiscardEnable = vk::False,
            rasterizer.polygonMode = vk::PolygonMode::eFill,
            rasterizer.cullMode = vk::CullModeFlagBits::eBack,
            rasterizer.frontFace = vk::FrontFace::eCounterClockwise,
            rasterizer.depthBiasEnable = vk::False
        ;
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling = { };
            multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1,
            multisampling.sampleShadingEnable = vk::False
        ;
        vk::PipelineDepthStencilStateCreateInfo depthStencil = {};
            depthStencil.depthTestEnable = vk::True,
            depthStencil.depthWriteEnable = vk::True,
            depthStencil.depthCompareOp = vk::CompareOp::eLess,
            depthStencil.depthBoundsTestEnable = vk::False,
            depthStencil.stencilTestEnable = vk::False
        ;
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = vk::False;

        vk::PipelineColorBlendStateCreateInfo colorBlending = { };
            colorBlending.logicOpEnable = vk::False,
            colorBlending.logicOp = vk::LogicOp::eCopy,
            colorBlending.attachmentCount = 1,
            colorBlending.pAttachments = &colorBlendAttachment
        ;

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState = { };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), dynamicState.pDynamicStates = dynamicStates.data();

        vk::DescriptorSetLayout setLayouts[] = {*descriptorSetLayoutGlobal, *descriptorSetLayoutMaterial};

        vk::PushConstantRange pushConstantRange = { };
            pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eFragment,
            pushConstantRange.offset = 0,
            pushConstantRange.size = sizeof(PushConstant)
        ;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = { };
          pipelineLayoutInfo.setLayoutCount = 2, pipelineLayoutInfo.pSetLayouts = setLayouts, pipelineLayoutInfo.pushConstantRangeCount = 1, pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::Format depthFormat = findDepthFormat();

        /* TASK01: Check the setup for dynamic rendering
         *
         * This new struct replaces what previously was the render pass in the pipeline creation.
         * Note how this structure is now linked in .pNext below, and .renderPass is not used.
         */
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo = { };
            pipelineRenderingCreateInfo.colorAttachmentCount = 1,
            pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChainImageFormat,
            pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat
        ;

        vk::GraphicsPipelineCreateInfo pipelineInfo = { };
            pipelineInfo.pNext = &pipelineRenderingCreateInfo,
            pipelineInfo.stageCount = 2,
            pipelineInfo.pStages = shaderStages,
            pipelineInfo.pVertexInputState = &vertexInputInfo,
            pipelineInfo.pInputAssemblyState = &inputAssembly,
            pipelineInfo.pViewportState = &viewportState,
            pipelineInfo.pRasterizationState = &rasterizer,
            pipelineInfo.pMultisampleState = &multisampling,
            pipelineInfo.pDepthStencilState = &depthStencil,
            pipelineInfo.pColorBlendState = &colorBlending,
            pipelineInfo.pDynamicState = &dynamicState,
            pipelineInfo.layout = pipelineLayout,
            pipelineInfo.renderPass = nullptr
        ;

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo = { };
            poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            poolInfo.queueFamilyIndex = graphicsIndex
        ;
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    static bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createTextureImage(const std::string& path) {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        vk::raii::Image textureImage = nullptr;
        vk::raii::DeviceMemory textureImageMemory = nullptr;

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        return std::make_pair(std::move(textureImage), std::move(textureImageMemory));
    }

    vk::raii::ImageView createTextureImageView(vk::raii::Image& textureImage) {
        return createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo = { };
            samplerInfo.magFilter = vk::Filter::eLinear,
            samplerInfo.minFilter = vk::Filter::eLinear,
            samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear,
            samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat,
            samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat,
            samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat,
            samplerInfo.mipLodBias = 0.0f,
            samplerInfo.anisotropyEnable = vk::True,
            samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            samplerInfo.compareEnable = vk::False,
            samplerInfo.compareOp = vk::CompareOp::eAlways
        ;
        textureSampler = vk::raii::Sampler(device, samplerInfo);
    }

    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
        vk::ImageViewCreateInfo viewInfo = {  };
            viewInfo.image = image,
            viewInfo.viewType = vk::ImageViewType::e2D,
            viewInfo.format = format,
            viewInfo.subresourceRange = { aspectFlags, 0, 1, 0, 1 }
        ;
        return vk::raii::ImageView(device, viewInfo);
    }

    void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
        vk::ImageCreateInfo imageInfo = { };
            imageInfo.imageType = vk::ImageType::e2D,
            imageInfo.format = format,
            imageInfo.extent = vk::Extent3D {width, height, 1},
            imageInfo.mipLevels = 1,
            imageInfo.arrayLayers = 1,
            imageInfo.samples = vk::SampleCountFlagBits::e1,
            imageInfo.tiling = tiling,
            imageInfo.usage = usage,
            imageInfo.sharingMode = vk::SharingMode::eExclusive,
            imageInfo.initialLayout = vk::ImageLayout::eUndefined
        ;
        image = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo = {};
            allocInfo.allocationSize = memRequirements.size,
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        ;
        imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        auto commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier = { };
            barrier.oldLayout = oldLayout,
            barrier.newLayout = newLayout,
            barrier.image = image,
            barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        ;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask =  vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask =  vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        commandBuffer->pipelineBarrier( sourceStage, destinationStage, {}, {}, nullptr, barrier );
        endSingleTimeCommands(*commandBuffer);
    }

    void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
        vk::BufferImageCopy region = { };
            region.bufferOffset = 0,
            region.bufferRowLength = 0,
            region.bufferImageHeight = 0,
            region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
            region.imageOffset = vk::Offset3D {0, 0, 0},
            region.imageExtent = vk::Extent3D {width, height, 1}
        ;
        commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
        endSingleTimeCommands(*commandBuffer);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> localMaterials;
        std::string warn, err;

        if (!LoadObj(&attrib, &shapes, &localMaterials, &warn, &err, MODEL_PATH.c_str(), MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")).c_str())) {
            throw std::runtime_error(warn + err);
        }

        size_t materialOffset = materials.size();
        size_t oldTextureCount = textureImageViews.size();

        materials.insert(materials.end(), localMaterials.begin(), localMaterials.end());

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        uint32_t indexOffset = 0;

        for (const auto& shape : shapes) {
            std::cout << "Loading mesh: " << shape.name << ": " << shape.mesh.indices.size()/3 << " triangles\n";

            uint32_t startOffset = indexOffset;
            uint32_t localMaxV = 0;

            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (index.normal_index >= 0) {
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 0.0f, 0.0f};
                }

                if (!uniqueVertices.contains(vertex)) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);

                indexOffset++;

                uint32_t vi;
                auto it = uniqueVertices.find(vertex);
                if (it != uniqueVertices.end()) {
                    vi = it->second;
                } else {
                    vi = static_cast<uint32_t>(vertices.size());
                    uniqueVertices[vertex] = vi;
                    vertices.push_back(vertex);
                }

                localMaxV = std::max(localMaxV, vi);
            }

            int localMaterialID = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[0];
            int globalMaterialID = (localMaterialID < 0) ? -1 : static_cast<int>(materialOffset + localMaterialID);

            uint32_t indexCount = indexOffset - startOffset;

            // Note that this is only valid for this particular MODEL_PATH
            bool alphaCut = (shape.name.find("nettle_plant") != std::string::npos);
            bool reflective = (shape.name.find("table") != std::string::npos);

            submeshes.push_back({
                .indexOffset = startOffset,
                .indexCount = indexCount,
                .materialID = globalMaterialID,
                .firstVertex = 0u,
                .maxVertex = localMaxV + 1,
                .alphaCut = alphaCut,
                .reflective = reflective
            });
        }

        for (size_t i = 0; i < localMaterials.size(); ++i) {
            const auto& material = localMaterials[i];

            if (!material.diffuse_texname.empty()) {
                std::string texturePath = MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")) + "/" + material.diffuse_texname;
                auto [img, mem] = createTextureImage(texturePath);
                textureImages.push_back(std::move(img));
                textureImageMemories.push_back(std::move(mem));
                textureImageViews.emplace_back(createTextureImageView(textureImages.back()));
            } else {
                std::cout << "No texture for material: " << material.name << std::endl;
            }
        }
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUVBuffer() {
        // Extract all texCoords into a separate vector
        std::vector<glm::vec2> uvs;
        uvs.reserve(vertices.size());
        for (auto& v: vertices) {
            uvs.push_back(v.texCoord);
        }

        vk::DeviceSize bufferSize = sizeof(uvs[0]) * uvs.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, uvs.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, uvBuffer, uvBufferMemory);

        copyBuffer(stagingBuffer, uvBuffer, bufferSize);
    }

    void createInstanceLUTBuffer() {
#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
        // TASK09: build a buffer to store the instance look-up table
        vk::DeviceSize bufferSize = sizeof(InstanceLUT) * instanceLUTs.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, instanceLUTs.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, instanceLUTBuffer, instanceLUTBufferMemory);

        copyBuffer(stagingBuffer, instanceLUTBuffer, bufferSize);
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createAccelerationStructures() {
#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
        vk::BufferDeviceAddressInfo vai = { }; vai.buffer = *vertexBuffer;
        vk::DeviceAddress vertexAddr = device.getBufferAddressKHR(vai);
        vk::BufferDeviceAddressInfo iai = { }; iai.buffer = *indexBuffer;
        vk::DeviceAddress indexAddr = device.getBufferAddressKHR(iai);

        instances.reserve(submeshes.size());
        blasBuffers.reserve(submeshes.size());
        blasMemories.reserve(submeshes.size());
        blasHandles.reserve(submeshes.size());

        vk::TransformMatrixKHR identity{};
        identity.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{1.f, 0.f, 0.f, 0.f},
            std::array<float,4>{0.f, 1.f, 0.f, 0.f},
            std::array<float,4>{0.f, 0.f, 1.f, 0.f}
        }};

        // TASK02: Build a bottom level acceleration structure for each submesh
        for (size_t i = 0; i < submeshes.size(); ++i) {
            const auto& submesh = submeshes[i];

            // Prepare the geometry data
            auto trianglesData = vk::AccelerationStructureGeometryTrianglesDataKHR { };
                trianglesData.vertexFormat = vk::Format::eR32G32B32Sfloat,
                trianglesData.vertexData = vertexAddr,
                trianglesData.vertexStride = sizeof(Vertex),
                trianglesData.maxVertex = submesh.maxVertex,
                trianglesData.indexType = vk::IndexType::eUint32,
                trianglesData.indexData = indexAddr + submesh.indexOffset * sizeof(uint32_t)
            ;

            vk::AccelerationStructureGeometryDataKHR geometryData(trianglesData);

            vk::AccelerationStructureGeometryKHR blasGeometry = { };
                blasGeometry.geometryType = vk::GeometryTypeKHR::eTriangles,
                blasGeometry.geometry = geometryData,
                blasGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque
            ;
#if LAB_TASK_LEVEL >= LAB_TASK_AS_OPAQUE_FLAG
            // TASK07
            blasGeometry.flags = (submesh.alphaCut) ? vk::GeometryFlagsKHR(0) : vk::GeometryFlagBitsKHR::eOpaque;
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_OPAQUE_FLAG

            vk::AccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo = { };
                blasBuildGeometryInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel,
                blasBuildGeometryInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                blasBuildGeometryInfo.geometryCount = 1,
                blasBuildGeometryInfo.pGeometries = &blasGeometry
            ;

            // Query the memory sizes that will be needed for this BLAS
            auto primitiveCount = static_cast<uint32_t>(submesh.indexCount / 3);

            vk::AccelerationStructureBuildSizesInfoKHR blasBuildSizes =
                device.getAccelerationStructureBuildSizesKHR(
                    vk::AccelerationStructureBuildTypeKHR::eDevice,
                    blasBuildGeometryInfo,
                    { primitiveCount }
            );

            // Create a scratch buffer for the BLAS, this will hold temporary data
            // during the build process
            vk::raii::Buffer scratchBuffer = nullptr;
            vk::raii::DeviceMemory scratchMemory = nullptr;
            createBuffer(blasBuildSizes.buildScratchSize,
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         scratchBuffer, scratchMemory);

            // Save the scratch buffer address in the build info structure
            vk::BufferDeviceAddressInfo scratchAddressInfo = { };
             scratchAddressInfo.buffer = *scratchBuffer;
            vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
            blasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

            // Create a buffer for the BLAS itself now that we now the required size
            vk::raii::Buffer blasBuffer = nullptr;
            vk::raii::DeviceMemory blasMemory = nullptr;
            blasBuffers.emplace_back(std::move(blasBuffer));
            blasMemories.emplace_back(std::move(blasMemory));
            createBuffer(blasBuildSizes.accelerationStructureSize,
                         vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         blasBuffers[i], blasMemories[i]);

            // Create and store the BLAS handle
            vk::AccelerationStructureCreateInfoKHR blasCreateInfo = { };
                blasCreateInfo.buffer = blasBuffers[i],
                blasCreateInfo.offset = 0,
                blasCreateInfo.size = blasBuildSizes.accelerationStructureSize,
                blasCreateInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel
            ;

            blasHandles.emplace_back(device.createAccelerationStructureKHR(blasCreateInfo));

            // Save the BLAS handle in the build info structure
            blasBuildGeometryInfo.dstAccelerationStructure = blasHandles[i];

            // Prepare the build range for the BLAS
            vk::AccelerationStructureBuildRangeInfoKHR blasRangeInfo = { };
                blasRangeInfo.primitiveCount = primitiveCount,
                blasRangeInfo.primitiveOffset = 0,
                blasRangeInfo.firstVertex = submesh.firstVertex,
                blasRangeInfo.transformOffset = 0
            ;

            // Build the BLAS
            auto cmd = beginSingleTimeCommands();
            cmd->buildAccelerationStructuresKHR({ blasBuildGeometryInfo }, { &blasRangeInfo });
            endSingleTimeCommands(*cmd);

            // TASK03: Create a BLAS instance for the TLAS
            vk::AccelerationStructureDeviceAddressInfoKHR addrInfo = { };
                addrInfo.accelerationStructure = *blasHandles[i];
            ;
            vk::DeviceAddress blasDeviceAddr = device.getAccelerationStructureAddressKHR(addrInfo);

            vk::AccelerationStructureInstanceKHR instance = { };
                instance.transform = identity,
                instance.mask = 0xFF,
                instance.accelerationStructureReference = blasDeviceAddr
            ;

            instances.push_back(instance);

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: store the instance look-up table entry
            instances[i].instanceCustomIndex = static_cast<uint32_t>(i);

            instanceLUTs.push_back({ static_cast<uint32_t>(submesh.materialID), submesh.indexOffset });
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
        }

        // TASK03: Prepare the instance data buffer
        vk::DeviceSize instBufferSize = sizeof(instances[0]) * instances.size();
        createBuffer(instBufferSize,
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                     instanceBuffer, instanceMemory);

        void *ptr = instanceMemory.mapMemory(0, instBufferSize);
        memcpy(ptr, instances.data(), instBufferSize);
        instanceMemory.unmapMemory();

        vk::BufferDeviceAddressInfo instanceAddrInfo = { }; instanceAddrInfo.buffer = instanceBuffer;
        vk::DeviceAddress instanceAddr = device.getBufferAddressKHR(instanceAddrInfo);

        // Prepare the geometry (instance) data
        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR { };
            instancesData.arrayOfPointers = vk::False,
            instancesData.data = instanceAddr
        ;

        vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

        vk::AccelerationStructureGeometryKHR tlasGeometry = { };
            tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances,
            tlasGeometry.geometry = geometryData
        ;

        vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo = { };
            tlasBuildGeometryInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel,
            tlasBuildGeometryInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            tlasBuildGeometryInfo.geometryCount = 1,
            tlasBuildGeometryInfo.pGeometries = &tlasGeometry
        ;

#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
        tlasBuildGeometryInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

        // Query the memory sizes that will be needed for this TLAS
        auto primitiveCount = static_cast<uint32_t>(instances.size());

        vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizes =
            device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice,
                tlasBuildGeometryInfo,
                { primitiveCount }
        );

        // Create a scratch buffer for the TLAS, this will hold temporary data
        // during the build process
        createBuffer(
            tlasBuildSizes.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasScratchBuffer, tlasScratchMemory
        );

        // Save the scratch buffer address in the build info structure
        vk::BufferDeviceAddressInfo scratchAddressInfo = { };
         scratchAddressInfo.buffer = *tlasScratchBuffer;
        vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
        tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

        // Create a buffer for the TLAS itself now that we now the required size
        createBuffer(
            tlasBuildSizes.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasBuffer, tlasMemory
        );

        // Create and store the TLAS handle
        vk::AccelerationStructureCreateInfoKHR tlasCreateInfo = { };
            tlasCreateInfo.buffer = tlasBuffer,
            tlasCreateInfo.offset = 0,
            tlasCreateInfo.size = tlasBuildSizes.accelerationStructureSize,
            tlasCreateInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;

        tlas = device.createAccelerationStructureKHR(tlasCreateInfo);

        // Save the TLAS handle in the build info structure
        tlasBuildGeometryInfo.dstAccelerationStructure = tlas;

         // Prepare the build range for the TLAS
         vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo = { };
             tlasRangeInfo.primitiveCount = primitiveCount,
             tlasRangeInfo.primitiveOffset = 0,
             tlasRangeInfo.firstVertex = 0,
             tlasRangeInfo.transformOffset = 0
         ;

        // Build the TLAS
        auto cmd = beginSingleTimeCommands();

        cmd->buildAccelerationStructuresKHR({ tlasBuildGeometryInfo }, { &tlasRangeInfo });

        endSingleTimeCommands(*cmd);
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
    }

    void createDescriptorPool() {
        std::array poolSize {
            vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize( vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize( vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 3), // indices, UVs, instance LUT
            vk::DescriptorPoolSize( vk::DescriptorType::eSampler, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize( vk::DescriptorType::eSampledImage, (uint32_t)materials.size())
        };
        vk::DescriptorPoolCreateInfo poolInfo = { };
            poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
            poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT + 1, // + 1 for bindless materials
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
            poolInfo.pPoolSizes = poolSize.data()
        ;
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        // Global descriptor sets (per frame)
        std::vector<vk::DescriptorSetLayout> globalLayouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayoutGlobal);

        vk::DescriptorSetAllocateInfo allocInfoGlobal = { };
            allocInfoGlobal.descriptorPool = descriptorPool,
            allocInfoGlobal.descriptorSetCount = static_cast<uint32_t>(globalLayouts.size()),
            allocInfoGlobal.pSetLayouts = globalLayouts.data()
        ;

        globalDescriptorSets.clear();
        globalDescriptorSets = device.allocateDescriptorSets(allocInfoGlobal);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
             // Uniform buffer
            vk::DescriptorBufferInfo bufferInfo = { };
                bufferInfo.buffer = uniformBuffers[i],
                bufferInfo.offset = 0,
                bufferInfo.range = sizeof(UniformBufferObject)
            ;

            vk::WriteDescriptorSet bufferWrite= { };
                bufferWrite.dstSet = globalDescriptorSets[i],
                bufferWrite.dstBinding = 0,
                bufferWrite.dstArrayElement = 0,
                bufferWrite.descriptorCount = 1,
                bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer,
                bufferWrite.pBufferInfo = &bufferInfo
            ;

#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
            // TASK04: define the acceleration structure descriptor.
            vk::WriteDescriptorSetAccelerationStructureKHR asInfo = { };
                asInfo.accelerationStructureCount = 1,
                asInfo.pAccelerationStructures = {&*tlas}
            ;

            vk::WriteDescriptorSet asWrite = { };
                asWrite.pNext = &asInfo,
                asWrite.dstSet = globalDescriptorSets[i],
                asWrite.dstBinding = 1,
                asWrite.dstArrayElement = 0,
                asWrite.descriptorCount = 1,
                asWrite.descriptorType = vk::DescriptorType::eAccelerationStructureKHR
            ;
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND

            // Indices SSBO
            vk::DescriptorBufferInfo indexBufferInfo = { };
                indexBufferInfo.buffer = indexBuffer,
                indexBufferInfo.offset = 0,
                indexBufferInfo.range = sizeof(uint32_t) * indices.size()
            ;

            vk::WriteDescriptorSet indexBufferWrite = { };
                indexBufferWrite.dstSet = globalDescriptorSets[i],
                indexBufferWrite.dstBinding = 2,
                indexBufferWrite.dstArrayElement = 0,
                indexBufferWrite.descriptorCount = 1,
                indexBufferWrite.descriptorType = vk::DescriptorType::eStorageBuffer,
                indexBufferWrite.pBufferInfo = &indexBufferInfo
            ;

            // UVs SSBO
            vk::DescriptorBufferInfo uvBufferInfo = { };
                uvBufferInfo.buffer = uvBuffer,
                uvBufferInfo.offset = 0,
                uvBufferInfo.range = sizeof(glm::vec2) * vertices.size()
            ;

            vk::WriteDescriptorSet uvBufferWrite = { };
                uvBufferWrite.dstSet = globalDescriptorSets[i],
                uvBufferWrite.dstBinding = 3,
                uvBufferWrite.dstArrayElement = 0,
                uvBufferWrite.descriptorCount = 1,
                uvBufferWrite.descriptorType = vk::DescriptorType::eStorageBuffer,
                uvBufferWrite.pBufferInfo = &uvBufferInfo
            ;

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: Instance LUT SSBO
            vk::DescriptorBufferInfo instanceLUTBufferInfo = { };
                instanceLUTBufferInfo.buffer = instanceLUTBuffer,
                instanceLUTBufferInfo.offset = 0,
                instanceLUTBufferInfo.range = sizeof(InstanceLUT) * instanceLUTs.size()
            ;

            vk::WriteDescriptorSet instanceLUTBufferWrite = { };
                instanceLUTBufferWrite.dstSet = globalDescriptorSets[i],
                instanceLUTBufferWrite.dstBinding = 4,
                instanceLUTBufferWrite.dstArrayElement = 0,
                instanceLUTBufferWrite.descriptorCount = 1,
                instanceLUTBufferWrite.descriptorType = vk::DescriptorType::eStorageBuffer,
                instanceLUTBufferWrite.pBufferInfo = &instanceLUTBufferInfo
            ;
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: Include the instance look-up table descriptor
            std::array<vk::WriteDescriptorSet, 5> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite, instanceLUTBufferWrite};
#elif LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
            // TASK04: Include the acceleration structure descriptor
            std::array<vk::WriteDescriptorSet, 4> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite};
#else
            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{bufferWrite, indexBufferWrite, uvBufferWrite};
#endif

            device.updateDescriptorSets(descriptorWrites, {});
        }

        // Material descriptor sets (per material)
        std::vector<uint32_t> variableCounts = { static_cast<uint32_t>(textureImageViews.size()) };
        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo = { };
            variableCountInfo.descriptorSetCount = 1,
            variableCountInfo.pDescriptorCounts = variableCounts.data()
        ;

        std::vector<vk::DescriptorSetLayout> layouts{ *descriptorSetLayoutMaterial };

        vk::DescriptorSetAllocateInfo allocInfo = { };
            allocInfo.pNext = &variableCountInfo,
            allocInfo.descriptorPool = descriptorPool,
            allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            allocInfo.pSetLayouts = layouts.data()
        ;

        materialDescriptorSets = device.allocateDescriptorSets(allocInfo);

        // Sampler
        vk::DescriptorImageInfo samplerInfo = { };
            samplerInfo.sampler = textureSampler
		;

        vk::WriteDescriptorSet samplerWrite = { };
            samplerWrite.dstSet = materialDescriptorSets[0],
            samplerWrite.dstBinding = 0,
            samplerWrite.dstArrayElement = 0,
            samplerWrite.descriptorCount = 1,
            samplerWrite.descriptorType = vk::DescriptorType::eSampler,
            samplerWrite.pImageInfo = &samplerInfo
        ;

        device.updateDescriptorSets({samplerWrite}, {});

        // Textures
        std::vector<vk::DescriptorImageInfo> imageInfos;
        imageInfos.reserve(textureImageViews.size());
        for (auto& iv : textureImageViews) {
            vk::DescriptorImageInfo imageInfo = { };
                imageInfo.imageView = iv,
                imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            ;
            imageInfos.push_back(imageInfo);
        }

        vk::WriteDescriptorSet materialWrite = { };
            materialWrite.dstSet = materialDescriptorSets[0],
            materialWrite.dstBinding = 1,
            materialWrite.dstArrayElement = 0,
            materialWrite.descriptorCount = static_cast<uint32_t>(imageInfos.size()),
            materialWrite.descriptorType = vk::DescriptorType::eSampledImage,
            materialWrite.pImageInfo = imageInfos.data()
        ;

        device.updateDescriptorSets({materialWrite}, {});
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo = { };
            bufferInfo.size = size,
            bufferInfo.usage = usage,
            bufferInfo.sharingMode = vk::SharingMode::eExclusive
        ;
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo = { };
            allocInfo.allocationSize = memRequirements.size,
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        ;
        vk::MemoryAllocateFlagsInfo allocFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            allocFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
            allocInfo.pNext = &allocFlagsInfo;
        }
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo = { };
            allocInfo.commandPool = commandPool,
            allocInfo.level = vk::CommandBufferLevel::ePrimary,
            allocInfo.commandBufferCount = 1
        ;
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

        vk::CommandBufferBeginInfo beginInfo = { };
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer->begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        vk::SubmitInfo submitInfo = { };
         submitInfo.commandBufferCount = 1, submitInfo.pCommandBuffers = &*commandBuffer;
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo = { };
         allocInfo.commandPool = commandPool, allocInfo.level = vk::CommandBufferLevel::ePrimary, allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo = {  };

        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        commandCopyBuffer.begin(beginInfo);
        vk::BufferCopy bc = { }; bc.size = size;
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, bc);
        commandCopyBuffer.end();
        vk::SubmitInfo submitInfo = { };
        submitInfo .commandBufferCount = 1, submitInfo.pCommandBuffers = &*commandCopyBuffer;
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo = { };
        allocInfo .commandPool = commandPool, allocInfo.level = vk::CommandBufferLevel::ePrimary,
                                                 allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame].begin({});
        // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},                                                     // srcAccessMask (no need to wait for previous operations)
            vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
            vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
            vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
        );
        // Transition depth image to depth attachment optimal layout
        vk::ImageMemoryBarrier2 depthBarrier = { };
            depthBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            depthBarrier.srcAccessMask = {},
            depthBarrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            depthBarrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            depthBarrier.oldLayout = vk::ImageLayout::eUndefined,
            depthBarrier.newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            depthBarrier.image = depthImage,
            depthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth,
            depthBarrier.subresourceRange.baseMipLevel = 0,
            depthBarrier.subresourceRange    .levelCount = 1,
            depthBarrier.subresourceRange    .baseArrayLayer = 0,
            depthBarrier.subresourceRange    .layerCount = 1
        ;
        vk::DependencyInfo depthDependencyInfo = { };
            depthDependencyInfo.dependencyFlags = {},
            depthDependencyInfo.imageMemoryBarrierCount = 1,
            depthDependencyInfo.pImageMemoryBarriers = &depthBarrier
        ;
        commandBuffers[currentFrame].pipelineBarrier2(depthDependencyInfo);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

        /* TASK01: Check the setup for dynamic rendering
         *
         * With dynamic rendering, we specify the image view and load/store operations directly
         * in the vk::RenderingAttachmentInfo structure.
         * This approach eliminates the need for explicit render pass and framebuffer objects,
         * simplifying the code and providing flexibility to change attachments at runtime.
         */

        vk::RenderingAttachmentInfo colorAttachmentInfo = { };
            colorAttachmentInfo.imageView = swapChainImageViews[imageIndex],
            colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear,
            colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore,
            colorAttachmentInfo.clearValue = clearColor
        ;

        vk::RenderingAttachmentInfo depthAttachmentInfo = {};
            depthAttachmentInfo.imageView = depthImageView,
            depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear,
            depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare,
            depthAttachmentInfo.clearValue = clearDepth
        ;

        // The vk::RenderingInfo structure combines these attachments with other rendering parameters.
        vk::RenderingInfo renderingInfo = { };
            renderingInfo.renderArea.offset = vk::Offset2D { 0, 0 };
            renderingInfo.renderArea.extent = swapChainExtent,
            renderingInfo.layerCount = 1,
            renderingInfo.colorAttachmentCount = 1,
            renderingInfo.pColorAttachments = &colorAttachmentInfo,
            renderingInfo.pDepthAttachment = &depthAttachmentInfo
        ;

        // Note: .beginRendering replaces the previous .beginRenderPass call.
        commandBuffers[currentFrame].beginRendering(renderingInfo);

        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
        commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffers[currentFrame].bindIndexBuffer( *indexBuffer, 0, vk::IndexType::eUint32 );
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *globalDescriptorSets[currentFrame], nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, *materialDescriptorSets[0], nullptr);

        for (auto& sub : submeshes) {
            // TASK09: Bindless resources
            PushConstant pushConstant = {
                .materialIndex = sub.materialID < 0 ? 0u : static_cast<uint32_t>(sub.materialID),
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
                .reflective = sub.reflective
#endif // LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
            };
            commandBuffers[currentFrame].pushConstants<PushConstant>(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pushConstant);

            commandBuffers[currentFrame].drawIndexed(sub.indexCount, 1, sub.indexOffset, 0, 0);
        }

        commandBuffers[currentFrame].endRendering();

        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,            // srcAccessMask
            {},                                                    // dstAccessMask
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,    // srcStage
            vk::PipelineStageFlagBits2::eBottomOfPipe              // dstStage
        );

        commandBuffers[currentFrame].end();
    }

    void transition_image_layout(
        uint32_t imageIndex,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask
        ) {
        vk::ImageMemoryBarrier2 barrier = { };
            barrier.srcStageMask = src_stage_mask,
            barrier.srcAccessMask = src_access_mask,
            barrier.dstStageMask = dst_stage_mask,
            barrier.dstAccessMask = dst_access_mask,
            barrier.oldLayout = old_layout,
            barrier.newLayout = new_layout,
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            barrier.image = swapChainImages[imageIndex],
            barrier.subresourceRange .aspectMask = vk::ImageAspectFlagBits::eColor,
            barrier.subresourceRange     .baseMipLevel = 0,
            barrier.subresourceRange     .levelCount = 1,
            barrier.subresourceRange     .baseArrayLayer = 0,
            barrier.subresourceRange     .layerCount = 1
        ;
        vk::DependencyInfo dependency_info = { };
            dependency_info.dependencyFlags = {},
            dependency_info.imageMemoryBarrierCount = 1,
            dependency_info.pImageMemoryBarriers = &barrier
        ;
        commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }

    void createSyncObjects() {
        presentCompleteSemaphore.clear();
        renderFinishedSemaphore.clear();
        inFlightFences.clear();

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
            renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
        }


        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::FenceCreateInfo info = { };
            info.flags = vk::FenceCreateFlagBits::eSignaled;
            inFlightFences.emplace_back(device, info);
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - startTime).count();

        auto eye = glm::vec3(2.0f, 2.0f, 2.0f);

        ubo.model = rotate(glm::mat4(1.0f), time * 0.1f * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(eye, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        ubo.cameraPos = eye;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
    void updateTopLevelAS(const glm::mat4 & model) {
        vk::TransformMatrixKHR tm{};
        auto &M = model;
        tm.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{M[0][0], M[1][0], M[2][0], M[3][0]},
            std::array<float,4>{M[0][1], M[1][1], M[2][1], M[3][1]},
            std::array<float,4>{M[0][2], M[1][2], M[2][2], M[3][2]}
        }};

        // TASK06: update the instances to use the new transform matrix.
        for (auto & instance : instances) {
            instance.setTransform(tm);
        }

        auto primitiveCount = static_cast<uint32_t>(instances.size());
        vk::DeviceSize instBufferSize = sizeof(instances[0]) * primitiveCount;

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(instBufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, instBufferSize);
        memcpy(dataStaging, instances.data(), instBufferSize);
        stagingBufferMemory.unmapMemory();

        copyBuffer(stagingBuffer, instanceBuffer, instBufferSize);

        vk::BufferDeviceAddressInfo instanceAddrInfo = { }; instanceAddrInfo.buffer = instanceBuffer;
        vk::DeviceAddress instanceAddr = device.getBufferAddressKHR(instanceAddrInfo);

        // Prepare the geometry (instance) data
        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR{ };
            instancesData.arrayOfPointers = vk::False,
            instancesData.data = instanceAddr
        ;

        vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

        vk::AccelerationStructureGeometryKHR tlasGeometry={ };
            tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances,
            tlasGeometry.geometry = geometryData
        ;

        // TASK06: Note the new parameters to re-build the TLAS in-place
        vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo = { };
            tlasBuildGeometryInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel,
            tlasBuildGeometryInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            tlasBuildGeometryInfo.mode = vk::BuildAccelerationStructureModeKHR::eUpdate,
            tlasBuildGeometryInfo.srcAccelerationStructure = tlas,
            tlasBuildGeometryInfo.dstAccelerationStructure = tlas,
            tlasBuildGeometryInfo.geometryCount = 1,
            tlasBuildGeometryInfo.pGeometries = &tlasGeometry
        ;

        vk::BufferDeviceAddressInfo scratchAddressInfo= { };
         scratchAddressInfo.buffer = *tlasScratchBuffer;
        vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
        tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

        // Prepare the build range for the TLAS
        vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo = { };
            tlasRangeInfo.primitiveCount = primitiveCount,
            tlasRangeInfo.primitiveOffset = 0,
            tlasRangeInfo.firstVertex = 0,
            tlasRangeInfo.transformOffset = 0
        ;

        // Re-build the TLAS
        auto cmd = beginSingleTimeCommands();

        // Pre-build barrier
        vk::MemoryBarrier preBarrier  = { };
            preBarrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead,
            preBarrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eAccelerationStructureWriteKHR
        ;

        cmd->pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eFragmentShader, // srcStageMask
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, // dstStageMask
            {}, // dependencyFlags
            preBarrier, // memoryBarriers
            {}, // bufferMemoryBarriers
            {} // imageMemoryBarriers
        );

        cmd->buildAccelerationStructuresKHR({ tlasBuildGeometryInfo }, { &tlasRangeInfo });

        // Post-build barrier
        vk::MemoryBarrier postBarrier = { };
            postBarrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
            postBarrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead
        ;

        cmd->pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, // srcStageMask
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eFragmentShader, // dstStageMask
            {}, // dependencyFlags
            postBarrier, // memoryBarriers
            {}, // bufferMemoryBarriers
            {} // imageMemoryBarriers
        );

        endSingleTimeCommands(*cmd);
    }
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

    void drawFrame() {
        while ( vk::Result::eTimeout == device.waitForFences( *inFlightFences[currentFrame], vk::True, UINT64_MAX ) )
            ;
        auto [result, imageIndex] = swapChain.acquireNextImage( UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr );

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        updateUniformBuffer(currentFrame);
#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
        // TASK06: Update the TLAS with the current model matrix
        updateTopLevelAS(ubo.model);
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

        device.resetFences(  *inFlightFences[currentFrame] );
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );
        /*const */vk::SubmitInfo submitInfo = { };
         submitInfo.waitSemaphoreCount = 1, submitInfo.pWaitSemaphores = &*presentCompleteSemaphore[semaphoreIndex],
                            submitInfo.pWaitDstStageMask = &waitDestinationStageMask, submitInfo.commandBufferCount = 1, submitInfo.pCommandBuffers = &*commandBuffers[currentFrame],
                            submitInfo.signalSemaphoreCount = 1, submitInfo.pSignalSemaphores = &*renderFinishedSemaphore[imageIndex];
        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);


        /*const */vk::PresentInfoKHR presentInfoKHR = { };
         presentInfoKHR.waitSemaphoreCount = 1, presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphore[imageIndex],
                                                presentInfoKHR.swapchainCount = 1,
                                                presentInfoKHR.pSwapchains = &*swapChain, presentInfoKHR.pImageIndices = &imageIndex;
        result = presentQueue.presentKHR(presentInfoKHR);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo = { };
        createInfo .codeSize = code.size(), createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
    }

    static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        const auto formatIt = std::ranges::find_if(availableFormats,
        [](const auto& format) {
            return format.format == vk::Format::eB8G8R8A8Srgb &&
                   format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
        return formatIt != availableFormats.end() ? formatIt->format : availableFormats[0].format;
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        return std::ranges::any_of(availablePresentModes,
            [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
            };
        }

    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
