## Objective-C Runtime Bindings
## Safe bindings for objc_msgSend with proper type casting
##
## v0.0.3: Runtime support for Metal bindings

when defined(macosx):
  # Selector registration
  proc sel_registerName*(name: cstring): pointer
    {.importc, header: "<objc/runtime.h>".}

  # objc_msgSend variants for different argument counts
  type
    ObjcMsgSend0* = proc(receiver: pointer, selector: pointer): pointer {.cdecl.}
    ObjcMsgSend1* = proc(receiver: pointer, selector: pointer, a1: pointer): pointer {.cdecl.}
    ObjcMsgSend1u* = proc(receiver: pointer, selector: pointer, a1: uint): pointer {.cdecl.}
    ObjcMsgSend2* = proc(receiver: pointer, selector: pointer, a1: pointer, a2: uint): pointer {.cdecl.}
    ObjcMsgSend2u* = proc(receiver: pointer, selector: pointer, a1: uint, a2: uint): pointer {.cdecl.}
    ObjcMsgSend3* = proc(receiver: pointer, selector: pointer, a1: pointer, a2: uint, a3: uint): pointer {.cdecl.}
    ObjcMsgSendStret* = proc(receiver: pointer, selector: pointer, a1: pointer): pointer {.cdecl.}

  var objc_msgSend_impl*: pointer
  {.emit: """
  #include <objc/message.h>
  N_LIB_PRIVATE void* objc_msgSend_impl = (void*)objc_msgSend;
  """.}

  # No-argument send
  template msgSend0*(recv, sel: pointer): pointer =
    cast[ObjcMsgSend0](objc_msgSend_impl)(recv, sel)

  # One pointer argument
  template msgSend1*(recv, sel: pointer, a1: pointer): pointer =
    cast[ObjcMsgSend1](objc_msgSend_impl)(recv, sel, a1)

  # One uint argument
  template msgSend1u*(recv, sel: pointer, a1: uint): pointer =
    cast[ObjcMsgSend1u](objc_msgSend_impl)(recv, sel, a1)

  # Two arguments (pointer, uint)
  template msgSend2*(recv, sel: pointer, a1: pointer, a2: uint): pointer =
    cast[ObjcMsgSend2](objc_msgSend_impl)(recv, sel, a1, a2)

  # Two uint arguments
  template msgSend2u*(recv, sel: pointer, a1: uint, a2: uint): pointer =
    cast[ObjcMsgSend2u](objc_msgSend_impl)(recv, sel, a1, a2)

  # Three arguments (pointer, uint, uint)
  template msgSend3*(recv, sel: pointer, a1: pointer, a2: uint, a3: uint): pointer =
    cast[ObjcMsgSend3](objc_msgSend_impl)(recv, sel, a1, a2, a3)

  # For struct returns
  template msgSendStret*(recv, sel: pointer, a1: pointer): pointer =
    cast[ObjcMsgSendStret](objc_msgSend_impl)(recv, sel, a1)

  # Additional variants for Metal command encoder
  type
    ObjcMsgSend1p* = proc(receiver: pointer, selector: pointer, a1: pointer): pointer {.cdecl.}
    ObjcMsgSend2s* = proc(receiver: pointer, selector: pointer, a1: MTLSizeType, a2: MTLSizeType): pointer {.cdecl.}
    MTLSizeType* = object
      width*, height*, depth*: uint

  # One pointer argument (for setComputePipelineState)
  template msgSend1p*(recv, sel: pointer, a1: pointer): pointer =
    cast[ObjcMsgSend1p](objc_msgSend_impl)(recv, sel, a1)

  # Two MTLSize arguments (for dispatchThreadgroups)
  template msgSend2s*(recv, sel: pointer, a1, a2: MTLSizeType): pointer =
    cast[ObjcMsgSend2s](objc_msgSend_impl)(recv, sel, a1, a2)

else:
  # Stub for non-macOS platforms
  proc sel_registerName*(name: cstring): pointer =
    nil

  template msgSend0*(recv, sel: pointer): pointer =
    nil

  template msgSend1*(recv, sel: pointer, a1: pointer): pointer =
    nil

  template msgSend1u*(recv, sel: pointer, a1: uint): pointer =
    nil

  template msgSend2*(recv, sel: pointer, a1: pointer, a2: uint): pointer =
    nil

  template msgSend2u*(recv, sel: pointer, a1: uint, a2: uint): pointer =
    nil

  template msgSend3*(recv, sel: pointer, a1: pointer, a2: uint, a3: uint): pointer =
    nil

  template msgSendStret*(recv, sel: pointer, a1: pointer): pointer =
    nil
