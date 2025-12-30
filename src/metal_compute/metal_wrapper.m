/**
 * Metal Framework C Wrapper
 *
 * This Objective-C file provides C-callable functions for Metal framework operations.
 * Nim code calls these C functions instead of directly using objc_msgSend.
 *
 * v0.0.4: Proper Metal runtime integration
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// ========== Device Functions ==========

/// Check if Metal is available on this system
int nmc_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil ? 1 : 0;
    }
}

/// Get the default Metal device
void* nmc_create_default_device(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (__bridge_retained void*)device;
    }
}

/// Release a Metal device
void nmc_release_device(void* device) {
    if (device != NULL) {
        @autoreleasepool {
            id<MTLDevice> mtlDevice = (__bridge_transfer id<MTLDevice>)device;
            mtlDevice = nil;  // Release
        }
    }
}

/// Get device name (returns allocated C string, caller must free)
const char* nmc_device_name(void* device) {
    if (device == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        NSString* name = mtlDevice.name;
        return strdup([name UTF8String]);
    }
}

/// Get device registry ID
uint64_t nmc_device_registry_id(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.registryID;
    }
}

/// Check if device is low power
int nmc_device_is_low_power(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.isLowPower ? 1 : 0;
    }
}

/// Check if device is headless
int nmc_device_is_headless(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.isHeadless ? 1 : 0;
    }
}

/// Check if device is removable
int nmc_device_is_removable(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.isRemovable ? 1 : 0;
    }
}

/// Check if device has unified memory
int nmc_device_has_unified_memory(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.hasUnifiedMemory ? 1 : 0;
    }
}

/// Get recommended max working set size
uint64_t nmc_device_recommended_max_working_set_size(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.recommendedMaxWorkingSetSize;
    }
}

/// Get max buffer length
uint64_t nmc_device_max_buffer_length(void* device) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return mtlDevice.maxBufferLength;
    }
}

/// Check if device supports GPU family
int nmc_device_supports_family(void* device, int family) {
    if (device == NULL) return 0;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return [mtlDevice supportsFamily:(MTLGPUFamily)family] ? 1 : 0;
    }
}

// ========== Buffer Functions ==========

/// Create a new buffer with specified length and options
void* nmc_create_buffer(void* device, uint64_t length, uint32_t options) {
    if (device == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:length
                                                      options:(MTLResourceOptions)options];
        return (__bridge_retained void*)buffer;
    }
}

/// Create a buffer with initial data
void* nmc_create_buffer_with_data(void* device, const void* data, uint64_t length, uint32_t options) {
    if (device == NULL || data == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLBuffer> buffer = [mtlDevice newBufferWithBytes:data
                                                      length:length
                                                     options:(MTLResourceOptions)options];
        return (__bridge_retained void*)buffer;
    }
}

/// Get buffer contents pointer
void* nmc_buffer_contents(void* buffer) {
    if (buffer == NULL) return NULL;
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        return mtlBuffer.contents;
    }
}

/// Get buffer length
uint64_t nmc_buffer_length(void* buffer) {
    if (buffer == NULL) return 0;
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        return mtlBuffer.length;
    }
}

/// Notify buffer was modified in range
void nmc_buffer_did_modify_range(void* buffer, uint64_t location, uint64_t length) {
    if (buffer == NULL) return;
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        [mtlBuffer didModifyRange:NSMakeRange(location, length)];
    }
}

/// Release a buffer
void nmc_release_buffer(void* buffer) {
    if (buffer != NULL) {
        @autoreleasepool {
            id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
            mtlBuffer = nil;
        }
    }
}

// ========== Command Queue Functions ==========

/// Create a new command queue
void* nmc_create_command_queue(void* device) {
    if (device == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
        return (__bridge_retained void*)queue;
    }
}

/// Release a command queue
void nmc_release_command_queue(void* queue) {
    if (queue != NULL) {
        @autoreleasepool {
            id<MTLCommandQueue> mtlQueue = (__bridge_transfer id<MTLCommandQueue>)queue;
            mtlQueue = nil;
        }
    }
}

// ========== Command Buffer Functions ==========

/// Create a new command buffer from queue
void* nmc_create_command_buffer(void* queue) {
    if (queue == NULL) return NULL;
    id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmdBuffer = [mtlQueue commandBuffer];
    // Explicitly retain the command buffer since it's autoreleased
    return (__bridge_retained void*)cmdBuffer;
}

/// Get command buffer status
int nmc_command_buffer_status(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 5;  // MTLCommandBufferStatusError
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    return (int)mtlCmdBuffer.status;
}

/// Commit command buffer
void nmc_command_buffer_commit(void* cmdBuffer) {
    if (cmdBuffer == NULL) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer commit];
}

/// Wait until command buffer is completed
void nmc_command_buffer_wait_until_completed(void* cmdBuffer) {
    if (cmdBuffer == NULL) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer waitUntilCompleted];
}

/// Wait until command buffer is scheduled
void nmc_command_buffer_wait_until_scheduled(void* cmdBuffer) {
    if (cmdBuffer == NULL) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer waitUntilScheduled];
}

/// Release a command buffer
void nmc_release_command_buffer(void* cmdBuffer) {
    if (cmdBuffer != NULL) {
        id<MTLCommandBuffer> mtlCmdBuffer = (__bridge_transfer id<MTLCommandBuffer>)cmdBuffer;
        (void)mtlCmdBuffer;  // Transfer ownership back to ARC which will release
    }
}

// ========== Compute Command Encoder Functions ==========

/// Create a compute command encoder
void* nmc_create_compute_encoder(void* cmdBuffer) {
    if (cmdBuffer == NULL) return NULL;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    id<MTLComputeCommandEncoder> encoder = [mtlCmdBuffer computeCommandEncoder];
    // Explicitly retain the encoder since it's autoreleased
    return (__bridge_retained void*)encoder;
}

/// Set buffer on encoder
void nmc_encoder_set_buffer(void* encoder, void* buffer, uint64_t offset, uint32_t index) {
    if (encoder == NULL || buffer == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    [mtlEncoder setBuffer:mtlBuffer offset:offset atIndex:index];
}

/// Set bytes on encoder
void nmc_encoder_set_bytes(void* encoder, const void* data, uint64_t length, uint32_t index) {
    if (encoder == NULL || data == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    [mtlEncoder setBytes:data length:length atIndex:index];
}

/// Set compute pipeline state on encoder
void nmc_encoder_set_pipeline_state(void* encoder, void* pipelineState) {
    if (encoder == NULL || pipelineState == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    id<MTLComputePipelineState> mtlPipeline = (__bridge id<MTLComputePipelineState>)pipelineState;
    [mtlEncoder setComputePipelineState:mtlPipeline];
}

/// Dispatch threadgroups
void nmc_encoder_dispatch_threadgroups(void* encoder,
                                        uint64_t gridW, uint64_t gridH, uint64_t gridD,
                                        uint64_t groupW, uint64_t groupH, uint64_t groupD) {
    if (encoder == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    MTLSize gridSize = MTLSizeMake(gridW, gridH, gridD);
    MTLSize groupSize = MTLSizeMake(groupW, groupH, groupD);
    [mtlEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
}

/// Dispatch threads (non-uniform)
void nmc_encoder_dispatch_threads(void* encoder,
                                   uint64_t threadsW, uint64_t threadsH, uint64_t threadsD,
                                   uint64_t groupW, uint64_t groupH, uint64_t groupD) {
    if (encoder == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    MTLSize threadsSize = MTLSizeMake(threadsW, threadsH, threadsD);
    MTLSize groupSize = MTLSizeMake(groupW, groupH, groupD);
    [mtlEncoder dispatchThreads:threadsSize threadsPerThreadgroup:groupSize];
}

/// End encoding
void nmc_encoder_end_encoding(void* encoder) {
    if (encoder == NULL) return;
    id<MTLComputeCommandEncoder> mtlEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    [mtlEncoder endEncoding];
}

/// Release encoder
void nmc_release_encoder(void* encoder) {
    if (encoder != NULL) {
        id<MTLComputeCommandEncoder> mtlEncoder = (__bridge_transfer id<MTLComputeCommandEncoder>)encoder;
        (void)mtlEncoder;  // Transfer ownership back to ARC which will release
    }
}

// ========== Shader and Pipeline Functions ==========

/// Compile Metal shader source code into a library
/// Returns the library handle or NULL on failure
/// errorOut will contain error message if compilation fails (caller must free)
void* nmc_compile_library(void* device, const char* source, char** errorOut) {
    if (device == NULL || source == NULL) {
        if (errorOut) *errorOut = strdup("Invalid device or source");
        return NULL;
    }

    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSString* sourceStr = [NSString stringWithUTF8String:source];
    NSError* error = nil;

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;

    id<MTLLibrary> library = [mtlDevice newLibraryWithSource:sourceStr
                                                     options:options
                                                       error:&error];

    if (error != nil || library == nil) {
        if (errorOut) {
            NSString* errorDesc = error ? [error localizedDescription] : @"Unknown compilation error";
            *errorOut = strdup([errorDesc UTF8String]);
        }
        return NULL;
    }

    return (__bridge_retained void*)library;
}

/// Load a precompiled Metal library from a file path
void* nmc_load_library(void* device, const char* path, char** errorOut) {
    if (device == NULL || path == NULL) {
        if (errorOut) *errorOut = strdup("Invalid device or path");
        return NULL;
    }

    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSString* pathStr = [NSString stringWithUTF8String:path];
    NSURL* url = [NSURL fileURLWithPath:pathStr];
    NSError* error = nil;

    id<MTLLibrary> library = [mtlDevice newLibraryWithURL:url error:&error];

    if (error != nil || library == nil) {
        if (errorOut) {
            NSString* errorDesc = error ? [error localizedDescription] : @"Failed to load library";
            *errorOut = strdup([errorDesc UTF8String]);
        }
        return NULL;
    }

    return (__bridge_retained void*)library;
}

/// Get the default library (compiled into the app bundle)
void* nmc_get_default_library(void* device) {
    if (device == NULL) return NULL;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLLibrary> library = [mtlDevice newDefaultLibrary];
    if (library == nil) return NULL;
    return (__bridge_retained void*)library;
}

/// Get a function from a library by name
void* nmc_get_function(void* library, const char* name) {
    if (library == NULL || name == NULL) return NULL;

    id<MTLLibrary> mtlLibrary = (__bridge id<MTLLibrary>)library;
    NSString* nameStr = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [mtlLibrary newFunctionWithName:nameStr];

    if (function == nil) return NULL;
    return (__bridge_retained void*)function;
}

/// Get function names from a library (returns newline-separated string, caller must free)
char* nmc_get_function_names(void* library) {
    if (library == NULL) return NULL;

    id<MTLLibrary> mtlLibrary = (__bridge id<MTLLibrary>)library;
    NSArray<NSString*>* names = [mtlLibrary functionNames];

    if (names == nil || names.count == 0) return strdup("");

    NSString* joined = [names componentsJoinedByString:@"\n"];
    return strdup([joined UTF8String]);
}

/// Create a compute pipeline state from a function
void* nmc_create_pipeline_state(void* device, void* function, char** errorOut) {
    if (device == NULL || function == NULL) {
        if (errorOut) *errorOut = strdup("Invalid device or function");
        return NULL;
    }

    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLFunction> mtlFunction = (__bridge id<MTLFunction>)function;
    NSError* error = nil;

    id<MTLComputePipelineState> pipeline = [mtlDevice newComputePipelineStateWithFunction:mtlFunction
                                                                                    error:&error];

    if (error != nil || pipeline == nil) {
        if (errorOut) {
            NSString* errorDesc = error ? [error localizedDescription] : @"Failed to create pipeline";
            *errorOut = strdup([errorDesc UTF8String]);
        }
        return NULL;
    }

    return (__bridge_retained void*)pipeline;
}

/// Get max total threads per threadgroup for a pipeline
uint64_t nmc_pipeline_max_threads_per_threadgroup(void* pipeline) {
    if (pipeline == NULL) return 0;
    id<MTLComputePipelineState> mtlPipeline = (__bridge id<MTLComputePipelineState>)pipeline;
    return (uint64_t)mtlPipeline.maxTotalThreadsPerThreadgroup;
}

/// Get thread execution width for a pipeline
uint64_t nmc_pipeline_thread_execution_width(void* pipeline) {
    if (pipeline == NULL) return 0;
    id<MTLComputePipelineState> mtlPipeline = (__bridge id<MTLComputePipelineState>)pipeline;
    return (uint64_t)mtlPipeline.threadExecutionWidth;
}

/// Get static threadgroup memory length for a pipeline
uint64_t nmc_pipeline_static_threadgroup_memory_length(void* pipeline) {
    if (pipeline == NULL) return 0;
    id<MTLComputePipelineState> mtlPipeline = (__bridge id<MTLComputePipelineState>)pipeline;
    return (uint64_t)mtlPipeline.staticThreadgroupMemoryLength;
}

/// Release a library
void nmc_release_library(void* library) {
    if (library != NULL) {
        id<MTLLibrary> mtlLibrary = (__bridge_transfer id<MTLLibrary>)library;
        (void)mtlLibrary;
    }
}

/// Release a function
void nmc_release_function(void* function) {
    if (function != NULL) {
        id<MTLFunction> mtlFunction = (__bridge_transfer id<MTLFunction>)function;
        (void)mtlFunction;
    }
}

/// Release a pipeline state
void nmc_release_pipeline_state(void* pipeline) {
    if (pipeline != NULL) {
        id<MTLComputePipelineState> mtlPipeline = (__bridge_transfer id<MTLComputePipelineState>)pipeline;
        (void)mtlPipeline;
    }
}

// ========== Async Execution Functions ==========

// Callback context structure for async operations
typedef struct {
    void (*callback)(void* context, int status);
    void* userContext;
} NMCCompletionContext;

/// Schedule a command buffer with a completion handler
/// The callback will be called when the command buffer completes
/// Returns 1 on success, 0 on failure
int nmc_command_buffer_add_completion_handler(void* cmdBuffer,
                                               void (*callback)(void* context, int status),
                                               void* userContext) {
    if (cmdBuffer == NULL || callback == NULL) return 0;

    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;

    // Copy the callback info to heap (so it persists until callback is called)
    NMCCompletionContext* ctx = (NMCCompletionContext*)malloc(sizeof(NMCCompletionContext));
    ctx->callback = callback;
    ctx->userContext = userContext;

    [mtlCmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        MTLCommandBufferStatus status = buffer.status;
        ctx->callback(ctx->userContext, (int)status);
        free(ctx);
    }];

    return 1;
}

/// Commit command buffer without waiting (async submission)
/// Returns 1 on success, 0 on failure
int nmc_command_buffer_commit_async(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer commit];
    return 1;
}

/// Check if command buffer execution has completed
/// Returns 1 if completed, 0 if still running
int nmc_command_buffer_is_completed(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    MTLCommandBufferStatus status = mtlCmdBuffer.status;
    return (status == MTLCommandBufferStatusCompleted ||
            status == MTLCommandBufferStatusError) ? 1 : 0;
}

/// Get command buffer GPU execution start time
/// Returns 0 if not available
double nmc_command_buffer_gpu_start_time(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0.0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    return mtlCmdBuffer.GPUStartTime;
}

/// Get command buffer GPU execution end time
/// Returns 0 if not available
double nmc_command_buffer_gpu_end_time(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0.0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    return mtlCmdBuffer.GPUEndTime;
}

/// Get command buffer kernel execution start time
/// Returns 0 if not available
double nmc_command_buffer_kernel_start_time(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0.0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    return mtlCmdBuffer.kernelStartTime;
}

/// Get command buffer kernel execution end time
/// Returns 0 if not available
double nmc_command_buffer_kernel_end_time(void* cmdBuffer) {
    if (cmdBuffer == NULL) return 0.0;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    return mtlCmdBuffer.kernelEndTime;
}

/// Get command buffer error message (if any)
/// Returns NULL if no error, caller must free the returned string
char* nmc_command_buffer_error_message(void* cmdBuffer) {
    if (cmdBuffer == NULL) return NULL;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    NSError* error = mtlCmdBuffer.error;
    if (error == nil) return NULL;
    return strdup([[error localizedDescription] UTF8String]);
}

/// Create command buffer with retained references (for double buffering)
void* nmc_create_command_buffer_retained(void* queue) {
    if (queue == NULL) return NULL;
    id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmdBuffer = [mtlQueue commandBufferWithUnretainedReferences];
    if (cmdBuffer == nil) {
        // Fallback to regular command buffer
        cmdBuffer = [mtlQueue commandBuffer];
    }
    return (__bridge_retained void*)cmdBuffer;
}

// ========== Event and Fence Functions for Synchronization ==========

/// Create a shared event for cross-command buffer synchronization
void* nmc_create_shared_event(void* device) {
    if (device == NULL) return NULL;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLSharedEvent> event = [mtlDevice newSharedEvent];
    if (event == nil) return NULL;
    return (__bridge_retained void*)event;
}

/// Get current value of shared event
uint64_t nmc_shared_event_value(void* event) {
    if (event == NULL) return 0;
    id<MTLSharedEvent> mtlEvent = (__bridge id<MTLSharedEvent>)event;
    return mtlEvent.signaledValue;
}

/// Encode wait for shared event in command buffer
void nmc_command_buffer_encode_wait_for_event(void* cmdBuffer, void* event, uint64_t value) {
    if (cmdBuffer == NULL || event == NULL) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    id<MTLSharedEvent> mtlEvent = (__bridge id<MTLSharedEvent>)event;
    [mtlCmdBuffer encodeWaitForEvent:mtlEvent value:value];
}

/// Encode signal for shared event in command buffer
void nmc_command_buffer_encode_signal_event(void* cmdBuffer, void* event, uint64_t value) {
    if (cmdBuffer == NULL || event == NULL) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    id<MTLSharedEvent> mtlEvent = (__bridge id<MTLSharedEvent>)event;
    [mtlCmdBuffer encodeSignalEvent:mtlEvent value:value];
}

/// Release shared event
void nmc_release_shared_event(void* event) {
    if (event != NULL) {
        id<MTLSharedEvent> mtlEvent = (__bridge_transfer id<MTLSharedEvent>)event;
        (void)mtlEvent;
    }
}

// ========== Utility Functions ==========

/// Free a C string allocated by this wrapper
void nmc_free_string(const char* str) {
    if (str != NULL) {
        free((void*)str);
    }
}
