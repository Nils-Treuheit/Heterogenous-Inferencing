ώσ	
η
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
delete_old_dirsbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Ύ
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718 

conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_35/kernel
~
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*'
_output_shapes
:*
dtype0
u
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_35/bias
n
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes	
:*
dtype0

conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_36/kernel

$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*(
_output_shapes
:*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:*
dtype0

conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:*
dtype0

conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:*
dtype0

conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_39/kernel

$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*(
_output_shapes
:*
dtype0
u
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_39/bias
n
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes	
:*
dtype0

conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_40/kernel

$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*(
_output_shapes
:*
dtype0
u
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_40/bias
n
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes	
:*
dtype0

conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_41/kernel

$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*(
_output_shapes
:*
dtype0
u
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_41/bias
n
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes	
:*
dtype0

conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_42/kernel

$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*(
_output_shapes
:*
dtype0
u
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_42/bias
n
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
β'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
value'B' B'
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
 
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
­

?layers
@layer_metrics

trainable_variables
Ametrics
Blayer_regularization_losses
regularization_losses
	variables
Cnon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Dlayers
Elayer_metrics
trainable_variables
Fmetrics
Glayer_regularization_losses
regularization_losses
	variables
Hnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_36/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_36/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Ilayers
Jlayer_metrics
trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
Mnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_37/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_37/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
Qlayer_regularization_losses
regularization_losses
	variables
Rnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_38/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_38/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
­

Slayers
Tlayer_metrics
#trainable_variables
Umetrics
Vlayer_regularization_losses
$regularization_losses
%	variables
Wnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_39/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_39/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
­

Xlayers
Ylayer_metrics
)trainable_variables
Zmetrics
[layer_regularization_losses
*regularization_losses
+	variables
\non_trainable_variables
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
­

]layers
^layer_metrics
/trainable_variables
_metrics
`layer_regularization_losses
0regularization_losses
1	variables
anon_trainable_variables
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
­

blayers
clayer_metrics
5trainable_variables
dmetrics
elayer_regularization_losses
6regularization_losses
7	variables
fnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
­

glayers
hlayer_metrics
;trainable_variables
imetrics
jlayer_regularization_losses
<regularization_losses
=	variables
knon_trainable_variables
?
0
1
2
3
4
5
6
7
	8
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
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:*
dtype0*
shape:
α
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_59594
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ϊ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_59995
΅
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/bias*
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_60053΄Γ
ΰJ

O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59646

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	D
(conv2d_36_conv2d_readvariableop_resource:8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	D
(conv2d_38_conv2d_readvariableop_resource:8
)conv2d_38_biasadd_readvariableop_resource:	D
(conv2d_39_conv2d_readvariableop_resource:8
)conv2d_39_biasadd_readvariableop_resource:	D
(conv2d_40_conv2d_readvariableop_resource:8
)conv2d_40_biasadd_readvariableop_resource:	D
(conv2d_41_conv2d_readvariableop_resource:8
)conv2d_41_biasadd_readvariableop_resource:	D
(conv2d_42_conv2d_readvariableop_resource:8
)conv2d_42_biasadd_readvariableop_resource:	
identity’ conv2d_35/BiasAdd/ReadVariableOp’conv2d_35/Conv2D/ReadVariableOp’ conv2d_36/BiasAdd/ReadVariableOp’conv2d_36/Conv2D/ReadVariableOp’ conv2d_37/BiasAdd/ReadVariableOp’conv2d_37/Conv2D/ReadVariableOp’ conv2d_38/BiasAdd/ReadVariableOp’conv2d_38/Conv2D/ReadVariableOp’ conv2d_39/BiasAdd/ReadVariableOp’conv2d_39/Conv2D/ReadVariableOp’ conv2d_40/BiasAdd/ReadVariableOp’conv2d_40/Conv2D/ReadVariableOp’ conv2d_41/BiasAdd/ReadVariableOp’conv2d_41/Conv2D/ReadVariableOp’ conv2d_42/BiasAdd/ReadVariableOp’conv2d_42/Conv2D/ReadVariableOp΄
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOpΊ
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
conv2d_35/Conv2D«
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp¨
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2
conv2d_35/BiasAdd΅
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOpΞ
conv2d_36/Conv2DConv2Dconv2d_35/BiasAdd:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
conv2d_36/Conv2D«
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp¨
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2
conv2d_36/BiasAdd΅
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOpΞ
conv2d_37/Conv2DConv2Dconv2d_36/BiasAdd:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
conv2d_37/Conv2D«
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp¨
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2
conv2d_37/BiasAdd΅
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_38/Conv2D/ReadVariableOpΞ
conv2d_38/Conv2DConv2Dconv2d_37/BiasAdd:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx*
paddingVALID*
strides
2
conv2d_38/Conv2D«
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp¨
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx2
conv2d_38/BiasAdd΅
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_39/Conv2D/ReadVariableOpΞ
conv2d_39/Conv2DConv2Dconv2d_38/BiasAdd:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv*
paddingVALID*
strides
2
conv2d_39/Conv2D«
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp¨
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv2
conv2d_39/BiasAdd΅
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_40/Conv2D/ReadVariableOpΞ
conv2d_40/Conv2DConv2Dconv2d_39/BiasAdd:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt*
paddingVALID*
strides
2
conv2d_40/Conv2D«
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp¨
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt2
conv2d_40/BiasAdd΅
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_41/Conv2D/ReadVariableOpΞ
conv2d_41/Conv2DConv2Dconv2d_40/BiasAdd:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr*
paddingVALID*
strides
2
conv2d_41/Conv2D«
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp¨
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr2
conv2d_41/BiasAdd΅
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_42/Conv2D/ReadVariableOpΞ
conv2d_42/Conv2DConv2Dconv2d_41/BiasAdd:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp*
paddingVALID*
strides
2
conv2d_42/Conv2D«
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp¨
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp2
conv2d_42/BiasAdd
IdentityIdentityconv2d_42/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs



D__inference_conv2d_37_layer_call_and_return_conditional_losses_59110

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:||
 
_user_specified_nameinputs



D__inference_conv2d_39_layer_call_and_return_conditional_losses_59142

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:vv2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:xx
 
_user_specified_nameinputs
ΰ
χ
4__inference_many_conv2d_stacked8_layer_call_fn_59735

inputs"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_591972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_59078

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_42_layer_call_fn_59924

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_591902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:rr: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:rr
 
_user_specified_nameinputs



D__inference_conv2d_41_layer_call_and_return_conditional_losses_59896

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:rr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:tt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:tt
 
_user_specified_nameinputs



D__inference_conv2d_38_layer_call_and_return_conditional_losses_59839

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:zz: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:zz
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_36_layer_call_fn_59810

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_590942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_40_layer_call_fn_59886

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_591582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:vv: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:vv
 
_user_specified_nameinputs
§
 
)__inference_conv2d_35_layer_call_fn_59791

inputs"
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_590782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_41_layer_call_fn_59905

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:rr*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_591742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:rr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:tt: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:tt
 
_user_specified_nameinputs
G
·

!__inference__traced_restore_60053
file_prefix<
!assignvariableop_conv2d_35_kernel:0
!assignvariableop_1_conv2d_35_bias:	?
#assignvariableop_2_conv2d_36_kernel:0
!assignvariableop_3_conv2d_36_bias:	?
#assignvariableop_4_conv2d_37_kernel:0
!assignvariableop_5_conv2d_37_bias:	?
#assignvariableop_6_conv2d_38_kernel:0
!assignvariableop_7_conv2d_38_bias:	?
#assignvariableop_8_conv2d_39_kernel:0
!assignvariableop_9_conv2d_39_bias:	@
$assignvariableop_10_conv2d_40_kernel:1
"assignvariableop_11_conv2d_40_bias:	@
$assignvariableop_12_conv2d_41_kernel:1
"assignvariableop_13_conv2d_41_bias:	@
$assignvariableop_14_conv2d_42_kernel:1
"assignvariableop_15_conv2d_42_bias:	
identity_17’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_36_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_36_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_37_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_37_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_38_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_38_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_39_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_39_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_40_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ͺ
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_40_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_41_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ͺ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_41_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_42_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ͺ
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_42_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΎ
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16±
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix



D__inference_conv2d_40_layer_call_and_return_conditional_losses_59877

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:vv: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:vv
 
_user_specified_nameinputs
£
η
#__inference_signature_wrapper_59594
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_590612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1



D__inference_conv2d_41_layer_call_and_return_conditional_losses_59174

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:rr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:tt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:tt
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_38_layer_call_fn_59848

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_591262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:zz: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:zz
 
_user_specified_nameinputs
ΰ
χ
4__inference_many_conv2d_stacked8_layer_call_fn_59772

inputs"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_593952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
ΰJ

O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59698

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	D
(conv2d_36_conv2d_readvariableop_resource:8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	D
(conv2d_38_conv2d_readvariableop_resource:8
)conv2d_38_biasadd_readvariableop_resource:	D
(conv2d_39_conv2d_readvariableop_resource:8
)conv2d_39_biasadd_readvariableop_resource:	D
(conv2d_40_conv2d_readvariableop_resource:8
)conv2d_40_biasadd_readvariableop_resource:	D
(conv2d_41_conv2d_readvariableop_resource:8
)conv2d_41_biasadd_readvariableop_resource:	D
(conv2d_42_conv2d_readvariableop_resource:8
)conv2d_42_biasadd_readvariableop_resource:	
identity’ conv2d_35/BiasAdd/ReadVariableOp’conv2d_35/Conv2D/ReadVariableOp’ conv2d_36/BiasAdd/ReadVariableOp’conv2d_36/Conv2D/ReadVariableOp’ conv2d_37/BiasAdd/ReadVariableOp’conv2d_37/Conv2D/ReadVariableOp’ conv2d_38/BiasAdd/ReadVariableOp’conv2d_38/Conv2D/ReadVariableOp’ conv2d_39/BiasAdd/ReadVariableOp’conv2d_39/Conv2D/ReadVariableOp’ conv2d_40/BiasAdd/ReadVariableOp’conv2d_40/Conv2D/ReadVariableOp’ conv2d_41/BiasAdd/ReadVariableOp’conv2d_41/Conv2D/ReadVariableOp’ conv2d_42/BiasAdd/ReadVariableOp’conv2d_42/Conv2D/ReadVariableOp΄
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOpΊ
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
conv2d_35/Conv2D«
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp¨
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2
conv2d_35/BiasAdd΅
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOpΞ
conv2d_36/Conv2DConv2Dconv2d_35/BiasAdd:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
conv2d_36/Conv2D«
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp¨
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2
conv2d_36/BiasAdd΅
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOpΞ
conv2d_37/Conv2DConv2Dconv2d_36/BiasAdd:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
conv2d_37/Conv2D«
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp¨
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2
conv2d_37/BiasAdd΅
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_38/Conv2D/ReadVariableOpΞ
conv2d_38/Conv2DConv2Dconv2d_37/BiasAdd:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx*
paddingVALID*
strides
2
conv2d_38/Conv2D«
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp¨
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx2
conv2d_38/BiasAdd΅
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_39/Conv2D/ReadVariableOpΞ
conv2d_39/Conv2DConv2Dconv2d_38/BiasAdd:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv*
paddingVALID*
strides
2
conv2d_39/Conv2D«
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp¨
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv2
conv2d_39/BiasAdd΅
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_40/Conv2D/ReadVariableOpΞ
conv2d_40/Conv2DConv2Dconv2d_39/BiasAdd:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt*
paddingVALID*
strides
2
conv2d_40/Conv2D«
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp¨
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt2
conv2d_40/BiasAdd΅
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_41/Conv2D/ReadVariableOpΞ
conv2d_41/Conv2DConv2Dconv2d_40/BiasAdd:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr*
paddingVALID*
strides
2
conv2d_41/Conv2D«
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp¨
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr2
conv2d_41/BiasAdd΅
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_42/Conv2D/ReadVariableOpΞ
conv2d_42/Conv2DConv2Dconv2d_41/BiasAdd:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp*
paddingVALID*
strides
2
conv2d_42/Conv2D«
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp¨
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp2
conv2d_42/BiasAdd
IdentityIdentityconv2d_42/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
γ
ψ
4__inference_many_conv2d_stacked8_layer_call_fn_59232
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_591972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
-
φ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59555
input_1*
conv2d_35_59514:
conv2d_35_59516:	+
conv2d_36_59519:
conv2d_36_59521:	+
conv2d_37_59524:
conv2d_37_59526:	+
conv2d_38_59529:
conv2d_38_59531:	+
conv2d_39_59534:
conv2d_39_59536:	+
conv2d_40_59539:
conv2d_40_59541:	+
conv2d_41_59544:
conv2d_41_59546:	+
conv2d_42_59549:
conv2d_42_59551:	
identity’!conv2d_35/StatefulPartitionedCall’!conv2d_36/StatefulPartitionedCall’!conv2d_37/StatefulPartitionedCall’!conv2d_38/StatefulPartitionedCall’!conv2d_39/StatefulPartitionedCall’!conv2d_40/StatefulPartitionedCall’!conv2d_41/StatefulPartitionedCall’!conv2d_42/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_59514conv2d_35_59516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_590782#
!conv2d_35/StatefulPartitionedCallΊ
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_59519conv2d_36_59521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_590942#
!conv2d_36/StatefulPartitionedCallΊ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_59524conv2d_37_59526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_591102#
!conv2d_37/StatefulPartitionedCallΊ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0conv2d_38_59529conv2d_38_59531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_591262#
!conv2d_38/StatefulPartitionedCallΊ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0conv2d_39_59534conv2d_39_59536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:vv*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_591422#
!conv2d_39/StatefulPartitionedCallΊ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0conv2d_40_59539conv2d_40_59541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_591582#
!conv2d_40/StatefulPartitionedCallΊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_59544conv2d_41_59546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:rr*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_591742#
!conv2d_41/StatefulPartitionedCallΊ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0conv2d_42_59549conv2d_42_59551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_591902#
!conv2d_42/StatefulPartitionedCall
IdentityIdentity*conv2d_42/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1



D__inference_conv2d_38_layer_call_and_return_conditional_losses_59126

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:zz: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:zz
 
_user_specified_nameinputs



D__inference_conv2d_42_layer_call_and_return_conditional_losses_59915

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:rr: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:rr
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_39_layer_call_fn_59867

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:vv*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_591422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:vv2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:xx: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:xx
 
_user_specified_nameinputs
-
υ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59395

inputs*
conv2d_35_59354:
conv2d_35_59356:	+
conv2d_36_59359:
conv2d_36_59361:	+
conv2d_37_59364:
conv2d_37_59366:	+
conv2d_38_59369:
conv2d_38_59371:	+
conv2d_39_59374:
conv2d_39_59376:	+
conv2d_40_59379:
conv2d_40_59381:	+
conv2d_41_59384:
conv2d_41_59386:	+
conv2d_42_59389:
conv2d_42_59391:	
identity’!conv2d_35/StatefulPartitionedCall’!conv2d_36/StatefulPartitionedCall’!conv2d_37/StatefulPartitionedCall’!conv2d_38/StatefulPartitionedCall’!conv2d_39/StatefulPartitionedCall’!conv2d_40/StatefulPartitionedCall’!conv2d_41/StatefulPartitionedCall’!conv2d_42/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_59354conv2d_35_59356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_590782#
!conv2d_35/StatefulPartitionedCallΊ
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_59359conv2d_36_59361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_590942#
!conv2d_36/StatefulPartitionedCallΊ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_59364conv2d_37_59366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_591102#
!conv2d_37/StatefulPartitionedCallΊ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0conv2d_38_59369conv2d_38_59371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_591262#
!conv2d_38/StatefulPartitionedCallΊ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0conv2d_39_59374conv2d_39_59376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:vv*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_591422#
!conv2d_39/StatefulPartitionedCallΊ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0conv2d_40_59379conv2d_40_59381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_591582#
!conv2d_40/StatefulPartitionedCallΊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_59384conv2d_41_59386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:rr*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_591742#
!conv2d_41/StatefulPartitionedCallΊ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0conv2d_42_59389conv2d_42_59391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_591902#
!conv2d_42/StatefulPartitionedCall
IdentityIdentity*conv2d_42/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
-
υ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59197

inputs*
conv2d_35_59079:
conv2d_35_59081:	+
conv2d_36_59095:
conv2d_36_59097:	+
conv2d_37_59111:
conv2d_37_59113:	+
conv2d_38_59127:
conv2d_38_59129:	+
conv2d_39_59143:
conv2d_39_59145:	+
conv2d_40_59159:
conv2d_40_59161:	+
conv2d_41_59175:
conv2d_41_59177:	+
conv2d_42_59191:
conv2d_42_59193:	
identity’!conv2d_35/StatefulPartitionedCall’!conv2d_36/StatefulPartitionedCall’!conv2d_37/StatefulPartitionedCall’!conv2d_38/StatefulPartitionedCall’!conv2d_39/StatefulPartitionedCall’!conv2d_40/StatefulPartitionedCall’!conv2d_41/StatefulPartitionedCall’!conv2d_42/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_59079conv2d_35_59081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_590782#
!conv2d_35/StatefulPartitionedCallΊ
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_59095conv2d_36_59097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_590942#
!conv2d_36/StatefulPartitionedCallΊ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_59111conv2d_37_59113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_591102#
!conv2d_37/StatefulPartitionedCallΊ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0conv2d_38_59127conv2d_38_59129*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_591262#
!conv2d_38/StatefulPartitionedCallΊ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0conv2d_39_59143conv2d_39_59145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:vv*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_591422#
!conv2d_39/StatefulPartitionedCallΊ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0conv2d_40_59159conv2d_40_59161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_591582#
!conv2d_40/StatefulPartitionedCallΊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_59175conv2d_41_59177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:rr*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_591742#
!conv2d_41/StatefulPartitionedCallΊ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0conv2d_42_59191conv2d_42_59193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_591902#
!conv2d_42/StatefulPartitionedCall
IdentityIdentity*conv2d_42/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_59782

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
γ
ψ
4__inference_many_conv2d_stacked8_layer_call_fn_59467
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_593952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1



D__inference_conv2d_36_layer_call_and_return_conditional_losses_59801

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs



D__inference_conv2d_39_layer_call_and_return_conditional_losses_59858

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:vv2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:xx
 
_user_specified_nameinputs



D__inference_conv2d_36_layer_call_and_return_conditional_losses_59094

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs
©,
λ
__inference__traced_save_59995
file_prefix/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesͺ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*π
_input_shapesή
Ϋ: ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
τi

 __inference__wrapped_model_59061
input_1X
=many_conv2d_stacked8_conv2d_35_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_35_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_36_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_36_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_37_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_37_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_38_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_38_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_39_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_39_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_40_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_40_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_41_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_41_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked8_conv2d_42_conv2d_readvariableop_resource:M
>many_conv2d_stacked8_conv2d_42_biasadd_readvariableop_resource:	
identity’5many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp’5many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOp’4many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOpσ
4many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOpϊ
%many_conv2d_stacked8/conv2d_35/Conv2DConv2Dinput_1<many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_35/Conv2Dκ
5many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_35/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_35/Conv2D:output:0=many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2(
&many_conv2d_stacked8/conv2d_35/BiasAddτ
4many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_36/Conv2DConv2D/many_conv2d_stacked8/conv2d_35/BiasAdd:output:0<many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_36/Conv2Dκ
5many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_36/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_36/Conv2D:output:0=many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2(
&many_conv2d_stacked8/conv2d_36/BiasAddτ
4many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_37/Conv2DConv2D/many_conv2d_stacked8/conv2d_36/BiasAdd:output:0<many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_37/Conv2Dκ
5many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_37/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_37/Conv2D:output:0=many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2(
&many_conv2d_stacked8/conv2d_37/BiasAddτ
4many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_38/Conv2DConv2D/many_conv2d_stacked8/conv2d_37/BiasAdd:output:0<many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_38/Conv2Dκ
5many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_38/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_38/Conv2D:output:0=many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:xx2(
&many_conv2d_stacked8/conv2d_38/BiasAddτ
4many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_39/Conv2DConv2D/many_conv2d_stacked8/conv2d_38/BiasAdd:output:0<many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_39/Conv2Dκ
5many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_39/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_39/Conv2D:output:0=many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:vv2(
&many_conv2d_stacked8/conv2d_39/BiasAddτ
4many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_40_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_40/Conv2DConv2D/many_conv2d_stacked8/conv2d_39/BiasAdd:output:0<many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_40/Conv2Dκ
5many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_40/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_40/Conv2D:output:0=many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt2(
&many_conv2d_stacked8/conv2d_40/BiasAddτ
4many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_41/Conv2DConv2D/many_conv2d_stacked8/conv2d_40/BiasAdd:output:0<many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_41/Conv2Dκ
5many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_41/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_41/Conv2D:output:0=many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:rr2(
&many_conv2d_stacked8/conv2d_41/BiasAddτ
4many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked8_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOp’
%many_conv2d_stacked8/conv2d_42/Conv2DConv2D/many_conv2d_stacked8/conv2d_41/BiasAdd:output:0<many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp*
paddingVALID*
strides
2'
%many_conv2d_stacked8/conv2d_42/Conv2Dκ
5many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked8_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOpό
&many_conv2d_stacked8/conv2d_42/BiasAddBiasAdd.many_conv2d_stacked8/conv2d_42/Conv2D:output:0=many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp2(
&many_conv2d_stacked8/conv2d_42/BiasAddϋ
IdentityIdentity/many_conv2d_stacked8/conv2d_42/BiasAdd:output:06^many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp6^many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOp5^many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2n
5many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_35/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_35/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_36/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_36/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_37/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_37/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_38/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_38/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_39/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_39/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_40/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_40/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_41/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_41/Conv2D/ReadVariableOp2n
5many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOp5many_conv2d_stacked8/conv2d_42/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOp4many_conv2d_stacked8/conv2d_42/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1



D__inference_conv2d_42_layer_call_and_return_conditional_losses_59190

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:pp2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:rr: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:rr
 
_user_specified_nameinputs
¦
‘
)__inference_conv2d_37_layer_call_fn_59829

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_591102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:||
 
_user_specified_nameinputs



D__inference_conv2d_40_layer_call_and_return_conditional_losses_59158

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:tt2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:vv: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:vv
 
_user_specified_nameinputs
-
φ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59511
input_1*
conv2d_35_59470:
conv2d_35_59472:	+
conv2d_36_59475:
conv2d_36_59477:	+
conv2d_37_59480:
conv2d_37_59482:	+
conv2d_38_59485:
conv2d_38_59487:	+
conv2d_39_59490:
conv2d_39_59492:	+
conv2d_40_59495:
conv2d_40_59497:	+
conv2d_41_59500:
conv2d_41_59502:	+
conv2d_42_59505:
conv2d_42_59507:	
identity’!conv2d_35/StatefulPartitionedCall’!conv2d_36/StatefulPartitionedCall’!conv2d_37/StatefulPartitionedCall’!conv2d_38/StatefulPartitionedCall’!conv2d_39/StatefulPartitionedCall’!conv2d_40/StatefulPartitionedCall’!conv2d_41/StatefulPartitionedCall’!conv2d_42/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_59470conv2d_35_59472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_590782#
!conv2d_35/StatefulPartitionedCallΊ
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_59475conv2d_36_59477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_590942#
!conv2d_36/StatefulPartitionedCallΊ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_59480conv2d_37_59482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_591102#
!conv2d_37/StatefulPartitionedCallΊ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0conv2d_38_59485conv2d_38_59487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_591262#
!conv2d_38/StatefulPartitionedCallΊ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0conv2d_39_59490conv2d_39_59492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:vv*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_591422#
!conv2d_39/StatefulPartitionedCallΊ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0conv2d_40_59495conv2d_40_59497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_591582#
!conv2d_40/StatefulPartitionedCallΊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_59500conv2d_41_59502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:rr*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_591742#
!conv2d_41/StatefulPartitionedCallΊ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0conv2d_42_59505conv2d_42_59507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_591902#
!conv2d_42/StatefulPartitionedCall
IdentityIdentity*conv2d_42/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall*
T0*'
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1



D__inference_conv2d_37_layer_call_and_return_conditional_losses_59820

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:||
 
_user_specified_nameinputs"ΜL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
<
input_11
serving_default_input_1:0=
	conv2d_420
StatefulPartitionedCall:0pptensorflow/serving/predict:έΖ
ΰq
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
*l&call_and_return_all_conditional_losses
m__call__
n_default_save_signature"ίm
_tf_keras_networkΓm{"name": "many_conv2d_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "many_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_38", "inbound_nodes": [[["conv2d_37", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_39", "inbound_nodes": [[["conv2d_38", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_40", "inbound_nodes": [[["conv2d_39", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_41", "inbound_nodes": [[["conv2d_40", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_42", "inbound_nodes": [[["conv2d_41", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_42", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "many_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_38", "inbound_nodes": [[["conv2d_37", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_39", "inbound_nodes": [[["conv2d_38", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_40", "inbound_nodes": [[["conv2d_39", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_41", "inbound_nodes": [[["conv2d_40", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_42", "inbound_nodes": [[["conv2d_41", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_42", 0, 0]]}}}
χ"τ
_tf_keras_input_layerΤ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ώ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"Ω	
_tf_keras_layerΏ	{"name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"ί	
_tf_keras_layerΕ	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 126, 126, 256]}}


kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*s&call_and_return_all_conditional_losses
t__call__"ί	
_tf_keras_layerΕ	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_36", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 124, 124, 256]}}


!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*u&call_and_return_all_conditional_losses
v__call__"β	
_tf_keras_layerΘ	{"name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_37", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 122, 122, 256]}}


'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
*w&call_and_return_all_conditional_losses
x__call__"β	
_tf_keras_layerΘ	{"name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_38", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120, 120, 256]}}


-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"β	
_tf_keras_layerΘ	{"name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_39", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 118, 118, 256]}}


3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
*{&call_and_return_all_conditional_losses
|__call__"β	
_tf_keras_layerΘ	{"name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_40", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 116, 116, 256]}}


9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*}&call_and_return_all_conditional_losses
~__call__"β	
_tf_keras_layerΘ	{"name": "conv2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_41", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 114, 114, 256]}}

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
Κ

?layers
@layer_metrics

trainable_variables
Ametrics
Blayer_regularization_losses
regularization_losses
	variables
Cnon_trainable_variables
m__call__
n_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
+:)2conv2d_35/kernel
:2conv2d_35/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Dlayers
Elayer_metrics
trainable_variables
Fmetrics
Glayer_regularization_losses
regularization_losses
	variables
Hnon_trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_36/kernel
:2conv2d_36/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Ilayers
Jlayer_metrics
trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
Mnon_trainable_variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_37/kernel
:2conv2d_37/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
Qlayer_regularization_losses
regularization_losses
	variables
Rnon_trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_38/kernel
:2conv2d_38/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
­

Slayers
Tlayer_metrics
#trainable_variables
Umetrics
Vlayer_regularization_losses
$regularization_losses
%	variables
Wnon_trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_39/kernel
:2conv2d_39/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
­

Xlayers
Ylayer_metrics
)trainable_variables
Zmetrics
[layer_regularization_losses
*regularization_losses
+	variables
\non_trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_40/kernel
:2conv2d_40/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
­

]layers
^layer_metrics
/trainable_variables
_metrics
`layer_regularization_losses
0regularization_losses
1	variables
anon_trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_41/kernel
:2conv2d_41/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
­

blayers
clayer_metrics
5trainable_variables
dmetrics
elayer_regularization_losses
6regularization_losses
7	variables
fnon_trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_42/kernel
:2conv2d_42/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
­

glayers
hlayer_metrics
;trainable_variables
imetrics
jlayer_regularization_losses
<regularization_losses
=	variables
knon_trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
_
0
1
2
3
4
5
6
7
	8"
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
2
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59646
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59698
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59511
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59555ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
4__inference_many_conv2d_stacked8_layer_call_fn_59232
4__inference_many_conv2d_stacked8_layer_call_fn_59735
4__inference_many_conv2d_stacked8_layer_call_fn_59772
4__inference_many_conv2d_stacked8_layer_call_fn_59467ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ί2ά
 __inference__wrapped_model_59061·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *'’$
"
input_1
ξ2λ
D__inference_conv2d_35_layer_call_and_return_conditional_losses_59782’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_35_layer_call_fn_59791’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_36_layer_call_and_return_conditional_losses_59801’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_36_layer_call_fn_59810’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_37_layer_call_and_return_conditional_losses_59820’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_37_layer_call_fn_59829’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_38_layer_call_and_return_conditional_losses_59839’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_38_layer_call_fn_59848’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_39_layer_call_and_return_conditional_losses_59858’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_39_layer_call_fn_59867’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_40_layer_call_and_return_conditional_losses_59877’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_40_layer_call_fn_59886’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_41_layer_call_and_return_conditional_losses_59896’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_41_layer_call_fn_59905’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_42_layer_call_and_return_conditional_losses_59915’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv2d_42_layer_call_fn_59924’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ΚBΗ
#__inference_signature_wrapper_59594input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
  
 __inference__wrapped_model_59061|!"'(-.349:1’.
'’$
"
input_1
ͺ "5ͺ2
0
	conv2d_42# 
	conv2d_42pp₯
D__inference_conv2d_35_layer_call_and_return_conditional_losses_59782]0’-
&’#
!
inputs
ͺ "%’"

0~~
 }
)__inference_conv2d_35_layer_call_fn_59791P0’-
&’#
!
inputs
ͺ "~~€
D__inference_conv2d_36_layer_call_and_return_conditional_losses_59801\/’,
%’"
 
inputs~~
ͺ "%’"

0||
 |
)__inference_conv2d_36_layer_call_fn_59810O/’,
%’"
 
inputs~~
ͺ "||€
D__inference_conv2d_37_layer_call_and_return_conditional_losses_59820\/’,
%’"
 
inputs||
ͺ "%’"

0zz
 |
)__inference_conv2d_37_layer_call_fn_59829O/’,
%’"
 
inputs||
ͺ "zz€
D__inference_conv2d_38_layer_call_and_return_conditional_losses_59839\!"/’,
%’"
 
inputszz
ͺ "%’"

0xx
 |
)__inference_conv2d_38_layer_call_fn_59848O!"/’,
%’"
 
inputszz
ͺ "xx€
D__inference_conv2d_39_layer_call_and_return_conditional_losses_59858\'(/’,
%’"
 
inputsxx
ͺ "%’"

0vv
 |
)__inference_conv2d_39_layer_call_fn_59867O'(/’,
%’"
 
inputsxx
ͺ "vv€
D__inference_conv2d_40_layer_call_and_return_conditional_losses_59877\-./’,
%’"
 
inputsvv
ͺ "%’"

0tt
 |
)__inference_conv2d_40_layer_call_fn_59886O-./’,
%’"
 
inputsvv
ͺ "tt€
D__inference_conv2d_41_layer_call_and_return_conditional_losses_59896\34/’,
%’"
 
inputstt
ͺ "%’"

0rr
 |
)__inference_conv2d_41_layer_call_fn_59905O34/’,
%’"
 
inputstt
ͺ "rr€
D__inference_conv2d_42_layer_call_and_return_conditional_losses_59915\9:/’,
%’"
 
inputsrr
ͺ "%’"

0pp
 |
)__inference_conv2d_42_layer_call_fn_59924O9:/’,
%’"
 
inputsrr
ͺ "ppΗ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59511t!"'(-.349:9’6
/’,
"
input_1
p 

 
ͺ "%’"

0pp
 Η
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59555t!"'(-.349:9’6
/’,
"
input_1
p

 
ͺ "%’"

0pp
 Ζ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59646s!"'(-.349:8’5
.’+
!
inputs
p 

 
ͺ "%’"

0pp
 Ζ
O__inference_many_conv2d_stacked8_layer_call_and_return_conditional_losses_59698s!"'(-.349:8’5
.’+
!
inputs
p

 
ͺ "%’"

0pp
 
4__inference_many_conv2d_stacked8_layer_call_fn_59232g!"'(-.349:9’6
/’,
"
input_1
p 

 
ͺ "pp
4__inference_many_conv2d_stacked8_layer_call_fn_59467g!"'(-.349:9’6
/’,
"
input_1
p

 
ͺ "pp
4__inference_many_conv2d_stacked8_layer_call_fn_59735f!"'(-.349:8’5
.’+
!
inputs
p 

 
ͺ "pp
4__inference_many_conv2d_stacked8_layer_call_fn_59772f!"'(-.349:8’5
.’+
!
inputs
p

 
ͺ "pp―
#__inference_signature_wrapper_59594!"'(-.349:<’9
’ 
2ͺ/
-
input_1"
input_1"5ͺ2
0
	conv2d_42# 
	conv2d_42pp