??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
?

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
 non_trainable_variables
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
?

!layers
"layer_metrics
trainable_variables
#metrics
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

&layers
'layer_metrics
trainable_variables
(metrics
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

+layers
,layer_metrics
trainable_variables
-metrics
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_34236
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_34511
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_34539??
?o
?
 __inference__wrapped_model_33975
input_1Y
?dilated_conv2d_stacked3_conv2d_8_conv2d_readvariableop_resource:N
@dilated_conv2d_stacked3_conv2d_8_biasadd_readvariableop_resource:Y
?dilated_conv2d_stacked3_conv2d_9_conv2d_readvariableop_resource:N
@dilated_conv2d_stacked3_conv2d_9_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked3_conv2d_10_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked3_conv2d_10_biasadd_readvariableop_resource:
identity??8dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp?7dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp?6dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp?7dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp?6dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp?
5dilated_conv2d_stacked3/conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5dilated_conv2d_stacked3/conv2d_8/Conv2D/dilation_rate?
Tdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2V
Tdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
Vdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2X
Vdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
Qdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2S
Qdilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
Ndilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2P
Ndilated_conv2d_stacked3/conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
Bdilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/block_shape?
?dilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?dilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/paddings?
6dilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinput_1Kdilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:0Hdilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@28
6dilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND?
6dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?dilated_conv2d_stacked3_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp?
'dilated_conv2d_stacked3/conv2d_8/Conv2DConv2D?dilated_conv2d_stacked3/conv2d_8/Conv2D/SpaceToBatchND:output:0>dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2)
'dilated_conv2d_stacked3/conv2d_8/Conv2D?
Bdilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/block_shape?
<dilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2>
<dilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/crops?
6dilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceND0dilated_conv2d_stacked3/conv2d_8/Conv2D:output:0Kdilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0Edilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||28
6dilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND?
7dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@dilated_conv2d_stacked3_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp?
(dilated_conv2d_stacked3/conv2d_8/BiasAddBiasAdd?dilated_conv2d_stacked3/conv2d_8/Conv2D/BatchToSpaceND:output:0?dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2*
(dilated_conv2d_stacked3/conv2d_8/BiasAdd?
5dilated_conv2d_stacked3/conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5dilated_conv2d_stacked3/conv2d_9/Conv2D/dilation_rate?
Tdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2V
Tdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
Vdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2X
Vdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
Qdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2S
Qdilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
Ndilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2P
Ndilated_conv2d_stacked3/conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
Bdilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/block_shape?
?dilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?dilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/paddings?
6dilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchND1dilated_conv2d_stacked3/conv2d_8/BiasAdd:output:0Kdilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:0Hdilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>28
6dilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND?
6dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?dilated_conv2d_stacked3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp?
'dilated_conv2d_stacked3/conv2d_9/Conv2DConv2D?dilated_conv2d_stacked3/conv2d_9/Conv2D/SpaceToBatchND:output:0>dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2)
'dilated_conv2d_stacked3/conv2d_9/Conv2D?
Bdilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/block_shape?
<dilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2>
<dilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/crops?
6dilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceND0dilated_conv2d_stacked3/conv2d_9/Conv2D:output:0Kdilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0Edilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx28
6dilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND?
7dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@dilated_conv2d_stacked3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp?
(dilated_conv2d_stacked3/conv2d_9/BiasAddBiasAdd?dilated_conv2d_stacked3/conv2d_9/Conv2D/BatchToSpaceND:output:0?dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2*
(dilated_conv2d_stacked3/conv2d_9/BiasAdd?
6dilated_conv2d_stacked3/conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked3/conv2d_10/Conv2D/dilation_rate?
Udilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2W
Udilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked3/conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchND1dilated_conv2d_stacked3/conv2d_9/BiasAdd:output:0Ldilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<29
7dilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked3/conv2d_10/Conv2DConv2D@dilated_conv2d_stacked3/conv2d_10/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2*
(dilated_conv2d_stacked3/conv2d_10/Conv2D?
Cdilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked3/conv2d_10/Conv2D:output:0Ldilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt29
7dilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked3/conv2d_10/BiasAddBiasAdd@dilated_conv2d_stacked3/conv2d_10/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2+
)dilated_conv2d_stacked3/conv2d_10/BiasAdd?
IdentityIdentity2dilated_conv2d_stacked3/conv2d_10/BiasAdd:output:09^dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp8^dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp7^dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp8^dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp7^dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2t
8dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp8dilated_conv2d_stacked3/conv2d_10/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp7dilated_conv2d_stacked3/conv2d_10/Conv2D/ReadVariableOp2r
7dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp7dilated_conv2d_stacked3/conv2d_8/BiasAdd/ReadVariableOp2p
6dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp6dilated_conv2d_stacked3/conv2d_8/Conv2D/ReadVariableOp2r
7dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp7dilated_conv2d_stacked3/conv2d_9/BiasAdd/ReadVariableOp2p
6dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp6dilated_conv2d_stacked3/conv2d_9/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
(__inference_conv2d_9_layer_call_fn_34440

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_340302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34079
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_340642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?R
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34346

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_8/Conv2D/dilation_rate?
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2>
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/SpaceToBatchND/block_shape?
'conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_8/Conv2D/SpaceToBatchND/paddings?
conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinputs3conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2 
conv2d_8/Conv2D/SpaceToBatchND?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2D'conv2d_8/Conv2D/SpaceToBatchND:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
conv2d_8/Conv2D?
*conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/BatchToSpaceND/block_shape?
$conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_8/Conv2D/BatchToSpaceND/crops?
conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_8/Conv2D:output:03conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2 
conv2d_8/Conv2D/BatchToSpaceND?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAdd'conv2d_8/Conv2D/BatchToSpaceND:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2
conv2d_8/BiasAdd?
conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_9/Conv2D/dilation_rate?
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2>
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/SpaceToBatchND/block_shape?
'conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_9/Conv2D/SpaceToBatchND/paddings?
conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_8/BiasAdd:output:03conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2 
conv2d_9/Conv2D/SpaceToBatchND?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D'conv2d_9/Conv2D/SpaceToBatchND:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
conv2d_9/Conv2D?
*conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/BatchToSpaceND/block_shape?
$conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_9/Conv2D/BatchToSpaceND/crops?
conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_9/Conv2D:output:03conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2 
conv2d_9/Conv2D/BatchToSpaceND?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAdd'conv2d_9/Conv2D/BatchToSpaceND:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2
conv2d_9/BiasAdd?
conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_10/Conv2D/dilation_rate?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/SpaceToBatchND/block_shape?
(conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_10/Conv2D/SpaceToBatchND/paddings?
conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_9/BiasAdd:output:04conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2!
conv2d_10/Conv2D/SpaceToBatchND?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D(conv2d_10/Conv2D/SpaceToBatchND:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
conv2d_10/Conv2D?
+conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/BatchToSpaceND/block_shape?
%conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_10/Conv2D/BatchToSpaceND/crops?
conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_10/Conv2D:output:04conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2!
conv2d_10/Conv2D/BatchToSpaceND?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAdd(conv2d_10/Conv2D/BatchToSpaceND:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2
conv2d_10/BiasAdd?
IdentityIdentityconv2d_10/BiasAdd:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_34236
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_339752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_34401

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
)__inference_conv2d_10_layer_call_fn_34470

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_340572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?
?
(__inference_conv2d_8_layer_call_fn_34410

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_340032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34147

inputs(
conv2d_8_34131:
conv2d_8_34133:(
conv2d_9_34136:
conv2d_9_34138:)
conv2d_10_34141:
conv2d_10_34143:
identity??!conv2d_10/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_34131conv2d_8_34133*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_340032"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_34136conv2d_9_34138*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_340302"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_34141conv2d_10_34143*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_340572#
!conv2d_10/StatefulPartitionedCall?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34363

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_340642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_34030

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34064

inputs(
conv2d_8_34004:
conv2d_8_34006:(
conv2d_9_34031:
conv2d_9_34033:)
conv2d_10_34058:
conv2d_10_34060:
identity??!conv2d_10/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_34004conv2d_8_34006*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_340032"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_34031conv2d_9_34033*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_340302"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_34058conv2d_10_34060*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_340572#
!conv2d_10/StatefulPartitionedCall?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34198
input_1(
conv2d_8_34182:
conv2d_8_34184:(
conv2d_9_34187:
conv2d_9_34189:)
conv2d_10_34192:
conv2d_10_34194:
identity??!conv2d_10/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8_34182conv2d_8_34184*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_340032"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_34187conv2d_9_34189*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_340302"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_34192conv2d_10_34194*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_340572#
!conv2d_10/StatefulPartitionedCall?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_34057

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_34461

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34380

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_341472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
__inference__traced_save_34511
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*_
_input_shapesN
L: ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34179
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_341472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
!__inference__traced_restore_34539
file_prefix:
 assignvariableop_conv2d_8_kernel:.
 assignvariableop_1_conv2d_8_bias:<
"assignvariableop_2_conv2d_9_kernel:.
 assignvariableop_3_conv2d_9_bias:=
#assignvariableop_4_conv2d_10_kernel:/
!assignvariableop_5_conv2d_10_bias:

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?R
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34291

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_8/Conv2D/dilation_rate?
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2>
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/SpaceToBatchND/block_shape?
'conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_8/Conv2D/SpaceToBatchND/paddings?
conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinputs3conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2 
conv2d_8/Conv2D/SpaceToBatchND?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2D'conv2d_8/Conv2D/SpaceToBatchND:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
conv2d_8/Conv2D?
*conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/BatchToSpaceND/block_shape?
$conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_8/Conv2D/BatchToSpaceND/crops?
conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_8/Conv2D:output:03conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2 
conv2d_8/Conv2D/BatchToSpaceND?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAdd'conv2d_8/Conv2D/BatchToSpaceND:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2
conv2d_8/BiasAdd?
conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_9/Conv2D/dilation_rate?
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2>
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/SpaceToBatchND/block_shape?
'conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_9/Conv2D/SpaceToBatchND/paddings?
conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_8/BiasAdd:output:03conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2 
conv2d_9/Conv2D/SpaceToBatchND?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D'conv2d_9/Conv2D/SpaceToBatchND:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
conv2d_9/Conv2D?
*conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/BatchToSpaceND/block_shape?
$conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_9/Conv2D/BatchToSpaceND/crops?
conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_9/Conv2D:output:03conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2 
conv2d_9/Conv2D/BatchToSpaceND?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAdd'conv2d_9/Conv2D/BatchToSpaceND:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2
conv2d_9/BiasAdd?
conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_10/Conv2D/dilation_rate?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/SpaceToBatchND/block_shape?
(conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_10/Conv2D/SpaceToBatchND/paddings?
conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_9/BiasAdd:output:04conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2!
conv2d_10/Conv2D/SpaceToBatchND?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D(conv2d_10/Conv2D/SpaceToBatchND:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
conv2d_10/Conv2D?
+conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/BatchToSpaceND/block_shape?
%conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_10/Conv2D/BatchToSpaceND/crops?
conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_10/Conv2D:output:04conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2!
conv2d_10/Conv2D/BatchToSpaceND?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAdd(conv2d_10/Conv2D/BatchToSpaceND:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2
conv2d_10/BiasAdd?
IdentityIdentityconv2d_10/BiasAdd:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_34003

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_34431

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34217
input_1(
conv2d_8_34201:
conv2d_8_34203:(
conv2d_9_34206:
conv2d_9_34208:)
conv2d_10_34211:
conv2d_10_34213:
identity??!conv2d_10/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8_34201conv2d_8_34203*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_340032"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_34206conv2d_9_34208*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_340302"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_34211conv2d_10_34213*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_340572#
!conv2d_10/StatefulPartitionedCall?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??: : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??<
	conv2d_10/
StatefulPartitionedCall:0tttensorflow/serving/predict:͖
?3
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*0&call_and_return_all_conditional_losses
1__call__
2_default_save_signature"?0
_tf_keras_network?0{"name": "dilated_conv2d_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "dilated_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_10", 0, 0]]}, "shared_object_id": 10, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dilated_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 9}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_10", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?



kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"?	
_tf_keras_layer?	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"?	
_tf_keras_layer?	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 124, 124, 12]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"?	
_tf_keras_layer?	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120, 120, 12]}}
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
?

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
 non_trainable_variables
1__call__
2_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
):'2conv2d_8/kernel
:2conv2d_8/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?

!layers
"layer_metrics
trainable_variables
#metrics
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_9/kernel
:2conv2d_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

&layers
'layer_metrics
trainable_variables
(metrics
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

+layers
,layer_metrics
trainable_variables
-metrics
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables
8__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34291
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34346
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34198
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34217?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34079
7__inference_dilated_conv2d_stacked3_layer_call_fn_34363
7__inference_dilated_conv2d_stacked3_layer_call_fn_34380
7__inference_dilated_conv2d_stacked3_layer_call_fn_34179?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_33975?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??
?2?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_34401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_8_layer_call_fn_34410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_34431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_9_layer_call_fn_34440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_34461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_10_layer_call_fn_34470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_34236input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_33975q
1?.
'?$
"?
input_1??
? "4?1
/
	conv2d_10"?
	conv2d_10tt?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_34461Z.?+
$?!
?
inputsxx
? "$?!
?
0tt
? z
)__inference_conv2d_10_layer_call_fn_34470M.?+
$?!
?
inputsxx
? "?tt?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_34401\
0?-
&?#
!?
inputs??
? "$?!
?
0||
? {
(__inference_conv2d_8_layer_call_fn_34410O
0?-
&?#
!?
inputs??
? "?||?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_34431Z.?+
$?!
?
inputs||
? "$?!
?
0xx
? y
(__inference_conv2d_9_layer_call_fn_34440M.?+
$?!
?
inputs||
? "?xx?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34198i
9?6
/?,
"?
input_1??
p 

 
? "$?!
?
0tt
? ?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34217i
9?6
/?,
"?
input_1??
p

 
? "$?!
?
0tt
? ?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34291h
8?5
.?+
!?
inputs??
p 

 
? "$?!
?
0tt
? ?
R__inference_dilated_conv2d_stacked3_layer_call_and_return_conditional_losses_34346h
8?5
.?+
!?
inputs??
p

 
? "$?!
?
0tt
? ?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34079\
9?6
/?,
"?
input_1??
p 

 
? "?tt?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34179\
9?6
/?,
"?
input_1??
p

 
? "?tt?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34363[
8?5
.?+
!?
inputs??
p 

 
? "?tt?
7__inference_dilated_conv2d_stacked3_layer_call_fn_34380[
8?5
.?+
!?
inputs??
p

 
? "?tt?
#__inference_signature_wrapper_34236|
<?9
? 
2?/
-
input_1"?
input_1??"4?1
/
	conv2d_10"?
	conv2d_10tt