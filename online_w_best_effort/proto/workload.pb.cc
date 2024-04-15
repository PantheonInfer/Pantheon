// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: workload.proto

#include "workload.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace protobuf_workload_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_workload_2eproto ::google::protobuf::internal::SCCInfo<0> scc_info_Workload;
}  // namespace protobuf_workload_2eproto
class WorkloadsDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Workloads>
      _instance;
} _Workloads_default_instance_;
class WorkloadDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Workload>
      _instance;
} _Workload_default_instance_;
namespace protobuf_workload_2eproto {
static void InitDefaultsWorkloads() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_Workloads_default_instance_;
    new (ptr) ::Workloads();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::Workloads::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_Workloads =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsWorkloads}, {
      &protobuf_workload_2eproto::scc_info_Workload.base,}};

static void InitDefaultsWorkload() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_Workload_default_instance_;
    new (ptr) ::Workload();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::Workload::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_Workload =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsWorkload}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_Workloads.base);
  ::google::protobuf::internal::InitSCC(&scc_info_Workload.base);
}

::google::protobuf::Metadata file_level_metadata[2];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workloads, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workloads, workload_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, model_name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, release_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, deadline_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, shape_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Workload, id_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::Workloads)},
  { 6, -1, sizeof(::Workload)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::_Workloads_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::_Workload_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "workload.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\016workload.proto\"(\n\tWorkloads\022\033\n\010workloa"
      "d\030\001 \003(\0132\t.Workload\"\\\n\010Workload\022\022\n\nmodel_"
      "name\030\001 \001(\t\022\017\n\007release\030\002 \001(\004\022\020\n\010deadline\030"
      "\003 \001(\004\022\r\n\005shape\030\005 \003(\003\022\n\n\002id\030\006 \001(\005b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 160);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "workload.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_workload_2eproto

// ===================================================================

void Workloads::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Workloads::kWorkloadFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Workloads::Workloads()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_workload_2eproto::scc_info_Workloads.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:Workloads)
}
Workloads::Workloads(const Workloads& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      workload_(from.workload_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:Workloads)
}

void Workloads::SharedCtor() {
}

Workloads::~Workloads() {
  // @@protoc_insertion_point(destructor:Workloads)
  SharedDtor();
}

void Workloads::SharedDtor() {
}

void Workloads::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* Workloads::descriptor() {
  ::protobuf_workload_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_workload_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Workloads& Workloads::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_workload_2eproto::scc_info_Workloads.base);
  return *internal_default_instance();
}


void Workloads::Clear() {
// @@protoc_insertion_point(message_clear_start:Workloads)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  workload_.Clear();
  _internal_metadata_.Clear();
}

bool Workloads::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:Workloads)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .Workload workload = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_workload()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:Workloads)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:Workloads)
  return false;
#undef DO_
}

void Workloads::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:Workloads)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .Workload workload = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->workload_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->workload(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:Workloads)
}

::google::protobuf::uint8* Workloads::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:Workloads)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .Workload workload = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->workload_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->workload(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Workloads)
  return target;
}

size_t Workloads::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Workloads)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .Workload workload = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->workload_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->workload(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Workloads::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:Workloads)
  GOOGLE_DCHECK_NE(&from, this);
  const Workloads* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Workloads>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:Workloads)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:Workloads)
    MergeFrom(*source);
  }
}

void Workloads::MergeFrom(const Workloads& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:Workloads)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  workload_.MergeFrom(from.workload_);
}

void Workloads::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:Workloads)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Workloads::CopyFrom(const Workloads& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Workloads)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Workloads::IsInitialized() const {
  return true;
}

void Workloads::Swap(Workloads* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Workloads::InternalSwap(Workloads* other) {
  using std::swap;
  CastToBase(&workload_)->InternalSwap(CastToBase(&other->workload_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata Workloads::GetMetadata() const {
  protobuf_workload_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_workload_2eproto::file_level_metadata[kIndexInFileMessages];
}


// ===================================================================

void Workload::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Workload::kModelNameFieldNumber;
const int Workload::kReleaseFieldNumber;
const int Workload::kDeadlineFieldNumber;
const int Workload::kShapeFieldNumber;
const int Workload::kIdFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Workload::Workload()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_workload_2eproto::scc_info_Workload.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:Workload)
}
Workload::Workload(const Workload& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      shape_(from.shape_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  model_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.model_name().size() > 0) {
    model_name_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.model_name_);
  }
  ::memcpy(&release_, &from.release_,
    static_cast<size_t>(reinterpret_cast<char*>(&id_) -
    reinterpret_cast<char*>(&release_)) + sizeof(id_));
  // @@protoc_insertion_point(copy_constructor:Workload)
}

void Workload::SharedCtor() {
  model_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&release_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&id_) -
      reinterpret_cast<char*>(&release_)) + sizeof(id_));
}

Workload::~Workload() {
  // @@protoc_insertion_point(destructor:Workload)
  SharedDtor();
}

void Workload::SharedDtor() {
  model_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void Workload::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* Workload::descriptor() {
  ::protobuf_workload_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_workload_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Workload& Workload::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_workload_2eproto::scc_info_Workload.base);
  return *internal_default_instance();
}


void Workload::Clear() {
// @@protoc_insertion_point(message_clear_start:Workload)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  shape_.Clear();
  model_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&release_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&id_) -
      reinterpret_cast<char*>(&release_)) + sizeof(id_));
  _internal_metadata_.Clear();
}

bool Workload::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:Workload)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string model_name = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_model_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->model_name().data(), static_cast<int>(this->model_name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "Workload.model_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint64 release = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(16u /* 16 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &release_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint64 deadline = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &deadline_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated int64 shape = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(42u /* 42 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, this->mutable_shape())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(40u /* 40 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 1, 42u, input, this->mutable_shape())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 id = 6;
      case 6: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(48u /* 48 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &id_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:Workload)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:Workload)
  return false;
#undef DO_
}

void Workload::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:Workload)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string model_name = 1;
  if (this->model_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->model_name().data(), static_cast<int>(this->model_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "Workload.model_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->model_name(), output);
  }

  // uint64 release = 2;
  if (this->release() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(2, this->release(), output);
  }

  // uint64 deadline = 3;
  if (this->deadline() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(3, this->deadline(), output);
  }

  // repeated int64 shape = 5;
  if (this->shape_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(5, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(static_cast< ::google::protobuf::uint32>(
        _shape_cached_byte_size_));
  }
  for (int i = 0, n = this->shape_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64NoTag(
      this->shape(i), output);
  }

  // int32 id = 6;
  if (this->id() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(6, this->id(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:Workload)
}

::google::protobuf::uint8* Workload::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:Workload)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string model_name = 1;
  if (this->model_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->model_name().data(), static_cast<int>(this->model_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "Workload.model_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->model_name(), target);
  }

  // uint64 release = 2;
  if (this->release() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(2, this->release(), target);
  }

  // uint64 deadline = 3;
  if (this->deadline() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(3, this->deadline(), target);
  }

  // repeated int64 shape = 5;
  if (this->shape_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      5,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast< ::google::protobuf::int32>(
            _shape_cached_byte_size_), target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteInt64NoTagToArray(this->shape_, target);
  }

  // int32 id = 6;
  if (this->id() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(6, this->id(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Workload)
  return target;
}

size_t Workload::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Workload)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated int64 shape = 5;
  {
    size_t data_size = ::google::protobuf::internal::WireFormatLite::
      Int64Size(this->shape_);
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast< ::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _shape_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // string model_name = 1;
  if (this->model_name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->model_name());
  }

  // uint64 release = 2;
  if (this->release() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt64Size(
        this->release());
  }

  // uint64 deadline = 3;
  if (this->deadline() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt64Size(
        this->deadline());
  }

  // int32 id = 6;
  if (this->id() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->id());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Workload::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:Workload)
  GOOGLE_DCHECK_NE(&from, this);
  const Workload* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Workload>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:Workload)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:Workload)
    MergeFrom(*source);
  }
}

void Workload::MergeFrom(const Workload& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:Workload)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  shape_.MergeFrom(from.shape_);
  if (from.model_name().size() > 0) {

    model_name_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.model_name_);
  }
  if (from.release() != 0) {
    set_release(from.release());
  }
  if (from.deadline() != 0) {
    set_deadline(from.deadline());
  }
  if (from.id() != 0) {
    set_id(from.id());
  }
}

void Workload::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:Workload)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Workload::CopyFrom(const Workload& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Workload)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Workload::IsInitialized() const {
  return true;
}

void Workload::Swap(Workload* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Workload::InternalSwap(Workload* other) {
  using std::swap;
  shape_.InternalSwap(&other->shape_);
  model_name_.Swap(&other->model_name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(release_, other->release_);
  swap(deadline_, other->deadline_);
  swap(id_, other->id_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata Workload::GetMetadata() const {
  protobuf_workload_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_workload_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::Workloads* Arena::CreateMaybeMessage< ::Workloads >(Arena* arena) {
  return Arena::CreateInternal< ::Workloads >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::Workload* Arena::CreateMaybeMessage< ::Workload >(Arena* arena) {
  return Arena::CreateInternal< ::Workload >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
