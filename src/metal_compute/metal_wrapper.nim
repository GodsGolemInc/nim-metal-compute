## Metal Framework C Wrapper Bindings
## Nim bindings for metal_wrapper.m
##
## v0.0.4: Proper Metal runtime integration via C wrapper

when defined(macosx):
  # Compile and link the Objective-C wrapper
  {.compile: "metal_wrapper.m".}
  {.passL: "-framework Metal".}
  {.passL: "-framework Foundation".}

  # ========== Device Functions ==========

  proc nmc_metal_available*(): cint
    {.importc, cdecl.}

  proc nmc_create_default_device*(): pointer
    {.importc, cdecl.}

  proc nmc_release_device*(device: pointer)
    {.importc, cdecl.}

  proc nmc_device_name*(device: pointer): cstring
    {.importc, cdecl.}

  proc nmc_device_registry_id*(device: pointer): uint64
    {.importc, cdecl.}

  proc nmc_device_is_low_power*(device: pointer): cint
    {.importc, cdecl.}

  proc nmc_device_is_headless*(device: pointer): cint
    {.importc, cdecl.}

  proc nmc_device_is_removable*(device: pointer): cint
    {.importc, cdecl.}

  proc nmc_device_has_unified_memory*(device: pointer): cint
    {.importc, cdecl.}

  proc nmc_device_recommended_max_working_set_size*(device: pointer): uint64
    {.importc, cdecl.}

  proc nmc_device_max_buffer_length*(device: pointer): uint64
    {.importc, cdecl.}

  proc nmc_device_supports_family*(device: pointer, family: cint): cint
    {.importc, cdecl.}

  # ========== Buffer Functions ==========

  proc nmc_create_buffer*(device: pointer, length: uint64, options: uint32): pointer
    {.importc, cdecl.}

  proc nmc_create_buffer_with_data*(device: pointer, data: pointer,
                                     length: uint64, options: uint32): pointer
    {.importc, cdecl.}

  proc nmc_buffer_contents*(buffer: pointer): pointer
    {.importc, cdecl.}

  proc nmc_buffer_length*(buffer: pointer): uint64
    {.importc, cdecl.}

  proc nmc_buffer_did_modify_range*(buffer: pointer, location: uint64, length: uint64)
    {.importc, cdecl.}

  proc nmc_release_buffer*(buffer: pointer)
    {.importc, cdecl.}

  # ========== Command Queue Functions ==========

  proc nmc_create_command_queue*(device: pointer): pointer
    {.importc, cdecl.}

  proc nmc_release_command_queue*(queue: pointer)
    {.importc, cdecl.}

  # ========== Command Buffer Functions ==========

  proc nmc_create_command_buffer*(queue: pointer): pointer
    {.importc, cdecl.}

  proc nmc_command_buffer_status*(cmdBuffer: pointer): cint
    {.importc, cdecl.}

  proc nmc_command_buffer_commit*(cmdBuffer: pointer)
    {.importc, cdecl.}

  proc nmc_command_buffer_wait_until_completed*(cmdBuffer: pointer)
    {.importc, cdecl.}

  proc nmc_command_buffer_wait_until_scheduled*(cmdBuffer: pointer)
    {.importc, cdecl.}

  proc nmc_release_command_buffer*(cmdBuffer: pointer)
    {.importc, cdecl.}

  # ========== Compute Command Encoder Functions ==========

  proc nmc_create_compute_encoder*(cmdBuffer: pointer): pointer
    {.importc, cdecl.}

  proc nmc_encoder_set_buffer*(encoder: pointer, buffer: pointer,
                                offset: uint64, index: uint32)
    {.importc, cdecl.}

  proc nmc_encoder_set_bytes*(encoder: pointer, data: pointer,
                               length: uint64, index: uint32)
    {.importc, cdecl.}

  proc nmc_encoder_set_pipeline_state*(encoder: pointer, pipelineState: pointer)
    {.importc, cdecl.}

  proc nmc_encoder_dispatch_threadgroups*(encoder: pointer,
                                           gridW, gridH, gridD: uint64,
                                           groupW, groupH, groupD: uint64)
    {.importc, cdecl.}

  proc nmc_encoder_dispatch_threads*(encoder: pointer,
                                      threadsW, threadsH, threadsD: uint64,
                                      groupW, groupH, groupD: uint64)
    {.importc, cdecl.}

  proc nmc_encoder_end_encoding*(encoder: pointer)
    {.importc, cdecl.}

  proc nmc_release_encoder*(encoder: pointer)
    {.importc, cdecl.}

  # ========== Shader and Pipeline Functions ==========

  proc nmc_compile_library*(device: pointer, source: cstring, errorOut: ptr cstring): pointer
    {.importc, cdecl.}

  proc nmc_load_library*(device: pointer, path: cstring, errorOut: ptr cstring): pointer
    {.importc, cdecl.}

  proc nmc_get_default_library*(device: pointer): pointer
    {.importc, cdecl.}

  proc nmc_get_function*(library: pointer, name: cstring): pointer
    {.importc, cdecl.}

  proc nmc_get_function_names*(library: pointer): cstring
    {.importc, cdecl.}

  proc nmc_create_pipeline_state*(device: pointer, function: pointer, errorOut: ptr cstring): pointer
    {.importc, cdecl.}

  proc nmc_pipeline_max_threads_per_threadgroup*(pipeline: pointer): uint64
    {.importc, cdecl.}

  proc nmc_pipeline_thread_execution_width*(pipeline: pointer): uint64
    {.importc, cdecl.}

  proc nmc_pipeline_static_threadgroup_memory_length*(pipeline: pointer): uint64
    {.importc, cdecl.}

  proc nmc_release_library*(library: pointer)
    {.importc, cdecl.}

  proc nmc_release_function*(function: pointer)
    {.importc, cdecl.}

  proc nmc_release_pipeline_state*(pipeline: pointer)
    {.importc, cdecl.}

  # ========== Async Execution Functions ==========

  type
    NMCCompletionCallback* = proc(context: pointer, status: cint) {.cdecl.}

  proc nmc_command_buffer_add_completion_handler*(cmdBuffer: pointer,
                                                    callback: NMCCompletionCallback,
                                                    userContext: pointer): cint
    {.importc, cdecl.}

  proc nmc_command_buffer_commit_async*(cmdBuffer: pointer): cint
    {.importc, cdecl.}

  proc nmc_command_buffer_is_completed*(cmdBuffer: pointer): cint
    {.importc, cdecl.}

  proc nmc_command_buffer_gpu_start_time*(cmdBuffer: pointer): cdouble
    {.importc, cdecl.}

  proc nmc_command_buffer_gpu_end_time*(cmdBuffer: pointer): cdouble
    {.importc, cdecl.}

  proc nmc_command_buffer_kernel_start_time*(cmdBuffer: pointer): cdouble
    {.importc, cdecl.}

  proc nmc_command_buffer_kernel_end_time*(cmdBuffer: pointer): cdouble
    {.importc, cdecl.}

  proc nmc_command_buffer_error_message*(cmdBuffer: pointer): cstring
    {.importc, cdecl.}

  proc nmc_create_command_buffer_retained*(queue: pointer): pointer
    {.importc, cdecl.}

  # ========== Event Functions for Synchronization ==========

  proc nmc_create_shared_event*(device: pointer): pointer
    {.importc, cdecl.}

  proc nmc_shared_event_value*(event: pointer): uint64
    {.importc, cdecl.}

  proc nmc_command_buffer_encode_wait_for_event*(cmdBuffer: pointer,
                                                   event: pointer,
                                                   value: uint64)
    {.importc, cdecl.}

  proc nmc_command_buffer_encode_signal_event*(cmdBuffer: pointer,
                                                 event: pointer,
                                                 value: uint64)
    {.importc, cdecl.}

  proc nmc_release_shared_event*(event: pointer)
    {.importc, cdecl.}

  # ========== Utility Functions ==========

  proc nmc_free_string*(str: cstring)
    {.importc, cdecl.}

else:
  # Stubs for non-macOS platforms

  proc nmc_metal_available*(): cint = 0
  proc nmc_create_default_device*(): pointer = nil
  proc nmc_release_device*(device: pointer) = discard
  proc nmc_device_name*(device: pointer): cstring = nil
  proc nmc_device_registry_id*(device: pointer): uint64 = 0
  proc nmc_device_is_low_power*(device: pointer): cint = 0
  proc nmc_device_is_headless*(device: pointer): cint = 0
  proc nmc_device_is_removable*(device: pointer): cint = 0
  proc nmc_device_has_unified_memory*(device: pointer): cint = 0
  proc nmc_device_recommended_max_working_set_size*(device: pointer): uint64 = 0
  proc nmc_device_max_buffer_length*(device: pointer): uint64 = 0
  proc nmc_device_supports_family*(device: pointer, family: cint): cint = 0
  proc nmc_create_buffer*(device: pointer, length: uint64, options: uint32): pointer = nil
  proc nmc_create_buffer_with_data*(device: pointer, data: pointer,
                                     length: uint64, options: uint32): pointer = nil
  proc nmc_buffer_contents*(buffer: pointer): pointer = nil
  proc nmc_buffer_length*(buffer: pointer): uint64 = 0
  proc nmc_buffer_did_modify_range*(buffer: pointer, location: uint64, length: uint64) = discard
  proc nmc_release_buffer*(buffer: pointer) = discard
  proc nmc_create_command_queue*(device: pointer): pointer = nil
  proc nmc_release_command_queue*(queue: pointer) = discard
  proc nmc_create_command_buffer*(queue: pointer): pointer = nil
  proc nmc_command_buffer_status*(cmdBuffer: pointer): cint = 5
  proc nmc_command_buffer_commit*(cmdBuffer: pointer) = discard
  proc nmc_command_buffer_wait_until_completed*(cmdBuffer: pointer) = discard
  proc nmc_command_buffer_wait_until_scheduled*(cmdBuffer: pointer) = discard
  proc nmc_release_command_buffer*(cmdBuffer: pointer) = discard
  proc nmc_create_compute_encoder*(cmdBuffer: pointer): pointer = nil
  proc nmc_encoder_set_buffer*(encoder: pointer, buffer: pointer,
                                offset: uint64, index: uint32) = discard
  proc nmc_encoder_set_bytes*(encoder: pointer, data: pointer,
                               length: uint64, index: uint32) = discard
  proc nmc_encoder_set_pipeline_state*(encoder: pointer, pipelineState: pointer) = discard
  proc nmc_encoder_dispatch_threadgroups*(encoder: pointer,
                                           gridW, gridH, gridD: uint64,
                                           groupW, groupH, groupD: uint64) = discard
  proc nmc_encoder_dispatch_threads*(encoder: pointer,
                                      threadsW, threadsH, threadsD: uint64,
                                      groupW, groupH, groupD: uint64) = discard
  proc nmc_encoder_end_encoding*(encoder: pointer) = discard
  proc nmc_release_encoder*(encoder: pointer) = discard
  proc nmc_compile_library*(device: pointer, source: cstring, errorOut: ptr cstring): pointer = nil
  proc nmc_load_library*(device: pointer, path: cstring, errorOut: ptr cstring): pointer = nil
  proc nmc_get_default_library*(device: pointer): pointer = nil
  proc nmc_get_function*(library: pointer, name: cstring): pointer = nil
  proc nmc_get_function_names*(library: pointer): cstring = nil
  proc nmc_create_pipeline_state*(device: pointer, function: pointer, errorOut: ptr cstring): pointer = nil
  proc nmc_pipeline_max_threads_per_threadgroup*(pipeline: pointer): uint64 = 0
  proc nmc_pipeline_thread_execution_width*(pipeline: pointer): uint64 = 0
  proc nmc_pipeline_static_threadgroup_memory_length*(pipeline: pointer): uint64 = 0
  proc nmc_release_library*(library: pointer) = discard
  proc nmc_release_function*(function: pointer) = discard
  proc nmc_release_pipeline_state*(pipeline: pointer) = discard
  # Async execution stubs
  type
    NMCCompletionCallback* = proc(context: pointer, status: cint) {.cdecl.}
  proc nmc_command_buffer_add_completion_handler*(cmdBuffer: pointer,
                                                    callback: NMCCompletionCallback,
                                                    userContext: pointer): cint = 0
  proc nmc_command_buffer_commit_async*(cmdBuffer: pointer): cint = 0
  proc nmc_command_buffer_is_completed*(cmdBuffer: pointer): cint = 1
  proc nmc_command_buffer_gpu_start_time*(cmdBuffer: pointer): cdouble = 0.0
  proc nmc_command_buffer_gpu_end_time*(cmdBuffer: pointer): cdouble = 0.0
  proc nmc_command_buffer_kernel_start_time*(cmdBuffer: pointer): cdouble = 0.0
  proc nmc_command_buffer_kernel_end_time*(cmdBuffer: pointer): cdouble = 0.0
  proc nmc_command_buffer_error_message*(cmdBuffer: pointer): cstring = nil
  proc nmc_create_command_buffer_retained*(queue: pointer): pointer = nil
  # Event stubs
  proc nmc_create_shared_event*(device: pointer): pointer = nil
  proc nmc_shared_event_value*(event: pointer): uint64 = 0
  proc nmc_command_buffer_encode_wait_for_event*(cmdBuffer: pointer,
                                                   event: pointer,
                                                   value: uint64) = discard
  proc nmc_command_buffer_encode_signal_event*(cmdBuffer: pointer,
                                                 event: pointer,
                                                 value: uint64) = discard
  proc nmc_release_shared_event*(event: pointer) = discard
  proc nmc_free_string*(str: cstring) = discard
