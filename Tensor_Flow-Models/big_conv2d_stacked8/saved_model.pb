ðê	
ç
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
¾
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718·

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:	*
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
:	*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:	*
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:	*
dtype0

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:	*
dtype0

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:	*
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:	*
dtype0

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:	*
dtype0

conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
â'
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
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
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
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_47210
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_47611
µ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias*
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
!__inference__traced_restore_47669ú»


ý
D__inference_conv2d_20_layer_call_and_return_conditional_losses_46710

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs


)__inference_conv2d_26_layer_call_fn_47540

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_468062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:VV	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:VV	
 
_user_specified_nameinputs


)__inference_conv2d_20_layer_call_fn_47426

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_467102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs


)__inference_conv2d_23_layer_call_fn_47483

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:bb	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_467582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:bb	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:hh	
 
_user_specified_nameinputs
õg
Ð
 __inference__wrapped_model_46677
input_1V
<big_conv2d_stacked8_conv2d_19_conv2d_readvariableop_resource:	K
=big_conv2d_stacked8_conv2d_19_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_20_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_20_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_21_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_21_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_22_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_22_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_23_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_23_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_24_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_24_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_25_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_25_biasadd_readvariableop_resource:	V
<big_conv2d_stacked8_conv2d_26_conv2d_readvariableop_resource:		K
=big_conv2d_stacked8_conv2d_26_biasadd_readvariableop_resource:	
identity¢4big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp¢4big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp¢3big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOpï
3big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype025
3big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOpö
$big_conv2d_stacked8/conv2d_19/Conv2DConv2Dinput_1;big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_19/Conv2Dæ
4big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_19/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_19/Conv2D:output:0<big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2'
%big_conv2d_stacked8/conv2d_19/BiasAddï
3big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_20/Conv2DConv2D.big_conv2d_stacked8/conv2d_19/BiasAdd:output:0;big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_20/Conv2Dæ
4big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_20/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_20/Conv2D:output:0<big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2'
%big_conv2d_stacked8/conv2d_20/BiasAddï
3big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_21/Conv2DConv2D.big_conv2d_stacked8/conv2d_20/BiasAdd:output:0;big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_21/Conv2Dæ
4big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_21/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_21/Conv2D:output:0<big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	2'
%big_conv2d_stacked8/conv2d_21/BiasAddï
3big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_22/Conv2DConv2D.big_conv2d_stacked8/conv2d_21/BiasAdd:output:0;big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_22/Conv2Dæ
4big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_22/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_22/Conv2D:output:0<big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	2'
%big_conv2d_stacked8/conv2d_22/BiasAddï
3big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_23/Conv2DConv2D.big_conv2d_stacked8/conv2d_22/BiasAdd:output:0;big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_23/Conv2Dæ
4big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_23/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_23/Conv2D:output:0<big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	2'
%big_conv2d_stacked8/conv2d_23/BiasAddï
3big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_24/Conv2DConv2D.big_conv2d_stacked8/conv2d_23/BiasAdd:output:0;big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_24/Conv2Dæ
4big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_24/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_24/Conv2D:output:0<big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	2'
%big_conv2d_stacked8/conv2d_24/BiasAddï
3big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_25/Conv2DConv2D.big_conv2d_stacked8/conv2d_24/BiasAdd:output:0;big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_25/Conv2Dæ
4big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_25/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_25/Conv2D:output:0<big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	2'
%big_conv2d_stacked8/conv2d_25/BiasAddï
3big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp<big_conv2d_stacked8_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype025
3big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOp
$big_conv2d_stacked8/conv2d_26/Conv2DConv2D.big_conv2d_stacked8/conv2d_25/BiasAdd:output:0;big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	*
paddingVALID*
strides
2&
$big_conv2d_stacked8/conv2d_26/Conv2Dæ
4big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp=big_conv2d_stacked8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp÷
%big_conv2d_stacked8/conv2d_26/BiasAddBiasAdd-big_conv2d_stacked8/conv2d_26/Conv2D:output:0<big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	2'
%big_conv2d_stacked8/conv2d_26/BiasAddé
IdentityIdentity.big_conv2d_stacked8/conv2d_26/BiasAdd:output:05^big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp5^big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp4^big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2l
4big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_19/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_19/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_20/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_20/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_21/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_21/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_22/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_22/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_23/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_23/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_24/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_24/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_25/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_25/Conv2D/ReadVariableOp2l
4big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp4big_conv2d_stacked8/conv2d_26/BiasAdd/ReadVariableOp2j
3big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOp3big_conv2d_stacked8/conv2d_26/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_21_layer_call_and_return_conditional_losses_46726

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:nn	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs


ý
D__inference_conv2d_21_layer_call_and_return_conditional_losses_47436

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:nn	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs


ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_46790

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:VV	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:\\	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:\\	
 
_user_specified_nameinputs


)__inference_conv2d_24_layer_call_fn_47502

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:\\	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_467742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:\\	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:bb	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:bb	
 
_user_specified_nameinputs


ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_47474

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:bb	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:hh	
 
_user_specified_nameinputs
£

)__inference_conv2d_19_layer_call_fn_47407

inputs!
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_466942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:zz	2

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
û+
ë
__inference__traced_save_47611
file_prefix/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
SaveV2/tensor_namesª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Ù
_input_shapesÇ
Ä: :	:	:		:	:		:	:		:	:		:	:		:	:		:	:		:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:	: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,	(
&
_output_shapes
:		: 


_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:

_output_shapes
: 
ã,
Þ
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47127
input_1)
conv2d_19_47086:	
conv2d_19_47088:	)
conv2d_20_47091:		
conv2d_20_47093:	)
conv2d_21_47096:		
conv2d_21_47098:	)
conv2d_22_47101:		
conv2d_22_47103:	)
conv2d_23_47106:		
conv2d_23_47108:	)
conv2d_24_47111:		
conv2d_24_47113:	)
conv2d_25_47116:		
conv2d_25_47118:	)
conv2d_26_47121:		
conv2d_26_47123:	
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_19_47086conv2d_19_47088*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_466942#
!conv2d_19/StatefulPartitionedCall¹
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_47091conv2d_20_47093*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_467102#
!conv2d_20/StatefulPartitionedCall¹
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_47096conv2d_21_47098*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:nn	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_467262#
!conv2d_21/StatefulPartitionedCall¹
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_47101conv2d_22_47103*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_467422#
!conv2d_22/StatefulPartitionedCall¹
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_47106conv2d_23_47108*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:bb	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_467582#
!conv2d_23/StatefulPartitionedCall¹
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_47111conv2d_24_47113*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:\\	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_467742#
!conv2d_24/StatefulPartitionedCall¹
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_47116conv2d_25_47118*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:VV	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_467902#
!conv2d_25/StatefulPartitionedCall¹
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_47121conv2d_26_47123*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_468062#
!conv2d_26/StatefulPartitionedCall
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_19_layer_call_and_return_conditional_losses_47398

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:zz	2

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


)__inference_conv2d_25_layer_call_fn_47521

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:VV	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_467902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:VV	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:\\	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:\\	
 
_user_specified_nameinputs


ý
D__inference_conv2d_26_layer_call_and_return_conditional_losses_47531

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:VV	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:VV	
 
_user_specified_nameinputs
È
à
3__inference_big_conv2d_stacked8_layer_call_fn_46848
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall¹
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
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_468132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

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
È
à
3__inference_big_conv2d_stacked8_layer_call_fn_47083
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall¹
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
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_470112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

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
 J
ý
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47314

inputsB
(conv2d_19_conv2d_readvariableop_resource:	7
)conv2d_19_biasadd_readvariableop_resource:	B
(conv2d_20_conv2d_readvariableop_resource:		7
)conv2d_20_biasadd_readvariableop_resource:	B
(conv2d_21_conv2d_readvariableop_resource:		7
)conv2d_21_biasadd_readvariableop_resource:	B
(conv2d_22_conv2d_readvariableop_resource:		7
)conv2d_22_biasadd_readvariableop_resource:	B
(conv2d_23_conv2d_readvariableop_resource:		7
)conv2d_23_biasadd_readvariableop_resource:	B
(conv2d_24_conv2d_readvariableop_resource:		7
)conv2d_24_biasadd_readvariableop_resource:	B
(conv2d_25_conv2d_readvariableop_resource:		7
)conv2d_25_biasadd_readvariableop_resource:	B
(conv2d_26_conv2d_readvariableop_resource:		7
)conv2d_26_biasadd_readvariableop_resource:	
identity¢ conv2d_19/BiasAdd/ReadVariableOp¢conv2d_19/Conv2D/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢ conv2d_25/BiasAdd/ReadVariableOp¢conv2d_25/Conv2D/ReadVariableOp¢ conv2d_26/BiasAdd/ReadVariableOp¢conv2d_26/Conv2D/ReadVariableOp³
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02!
conv2d_19/Conv2D/ReadVariableOp¹
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
conv2d_19/Conv2Dª
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp§
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2
conv2d_19/BiasAdd³
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_20/Conv2D/ReadVariableOpÍ
conv2d_20/Conv2DConv2Dconv2d_19/BiasAdd:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
conv2d_20/Conv2Dª
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp§
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2
conv2d_20/BiasAdd³
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_21/Conv2D/ReadVariableOpÍ
conv2d_21/Conv2DConv2Dconv2d_20/BiasAdd:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	*
paddingVALID*
strides
2
conv2d_21/Conv2Dª
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp§
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	2
conv2d_21/BiasAdd³
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_22/Conv2D/ReadVariableOpÍ
conv2d_22/Conv2DConv2Dconv2d_21/BiasAdd:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	*
paddingVALID*
strides
2
conv2d_22/Conv2Dª
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp§
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	2
conv2d_22/BiasAdd³
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_23/Conv2D/ReadVariableOpÍ
conv2d_23/Conv2DConv2Dconv2d_22/BiasAdd:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	*
paddingVALID*
strides
2
conv2d_23/Conv2Dª
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp§
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	2
conv2d_23/BiasAdd³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_24/Conv2D/ReadVariableOpÍ
conv2d_24/Conv2DConv2Dconv2d_23/BiasAdd:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	*
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp§
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	2
conv2d_24/BiasAdd³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_25/Conv2D/ReadVariableOpÍ
conv2d_25/Conv2DConv2Dconv2d_24/BiasAdd:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	*
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp§
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	2
conv2d_25/BiasAdd³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_26/Conv2D/ReadVariableOpÍ
conv2d_26/Conv2DConv2Dconv2d_25/BiasAdd:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp§
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	2
conv2d_26/BiasAdd
IdentityIdentityconv2d_26/BiasAdd:output:0!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
à,
Ý
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47011

inputs)
conv2d_19_46970:	
conv2d_19_46972:	)
conv2d_20_46975:		
conv2d_20_46977:	)
conv2d_21_46980:		
conv2d_21_46982:	)
conv2d_22_46985:		
conv2d_22_46987:	)
conv2d_23_46990:		
conv2d_23_46992:	)
conv2d_24_46995:		
conv2d_24_46997:	)
conv2d_25_47000:		
conv2d_25_47002:	)
conv2d_26_47005:		
conv2d_26_47007:	
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_46970conv2d_19_46972*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_466942#
!conv2d_19/StatefulPartitionedCall¹
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_46975conv2d_20_46977*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_467102#
!conv2d_20/StatefulPartitionedCall¹
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_46980conv2d_21_46982*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:nn	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_467262#
!conv2d_21/StatefulPartitionedCall¹
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_46985conv2d_22_46987*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_467422#
!conv2d_22/StatefulPartitionedCall¹
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_46990conv2d_23_46992*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:bb	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_467582#
!conv2d_23/StatefulPartitionedCall¹
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_46995conv2d_24_46997*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:\\	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_467742#
!conv2d_24/StatefulPartitionedCall¹
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_47000conv2d_25_47002*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:VV	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_467902#
!conv2d_25/StatefulPartitionedCall¹
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_47005conv2d_26_47007*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_468062#
!conv2d_26/StatefulPartitionedCall
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_47512

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:VV	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:\\	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:\\	
 
_user_specified_nameinputs


ý
D__inference_conv2d_20_layer_call_and_return_conditional_losses_47417

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs


)__inference_conv2d_22_layer_call_fn_47464

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_467422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:hh	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:nn	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:nn	
 
_user_specified_nameinputs

Ð
#__inference_signature_wrapper_47210
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall
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
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_466772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

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


ý
D__inference_conv2d_19_layer_call_and_return_conditional_losses_46694

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:zz	2

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


ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_46758

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:bb	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:hh	
 
_user_specified_nameinputs
Å
ß
3__inference_big_conv2d_stacked8_layer_call_fn_47388

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall¸
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
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_470112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

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
 J
ý
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47262

inputsB
(conv2d_19_conv2d_readvariableop_resource:	7
)conv2d_19_biasadd_readvariableop_resource:	B
(conv2d_20_conv2d_readvariableop_resource:		7
)conv2d_20_biasadd_readvariableop_resource:	B
(conv2d_21_conv2d_readvariableop_resource:		7
)conv2d_21_biasadd_readvariableop_resource:	B
(conv2d_22_conv2d_readvariableop_resource:		7
)conv2d_22_biasadd_readvariableop_resource:	B
(conv2d_23_conv2d_readvariableop_resource:		7
)conv2d_23_biasadd_readvariableop_resource:	B
(conv2d_24_conv2d_readvariableop_resource:		7
)conv2d_24_biasadd_readvariableop_resource:	B
(conv2d_25_conv2d_readvariableop_resource:		7
)conv2d_25_biasadd_readvariableop_resource:	B
(conv2d_26_conv2d_readvariableop_resource:		7
)conv2d_26_biasadd_readvariableop_resource:	
identity¢ conv2d_19/BiasAdd/ReadVariableOp¢conv2d_19/Conv2D/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢ conv2d_25/BiasAdd/ReadVariableOp¢conv2d_25/Conv2D/ReadVariableOp¢ conv2d_26/BiasAdd/ReadVariableOp¢conv2d_26/Conv2D/ReadVariableOp³
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02!
conv2d_19/Conv2D/ReadVariableOp¹
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
conv2d_19/Conv2Dª
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp§
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2
conv2d_19/BiasAdd³
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_20/Conv2D/ReadVariableOpÍ
conv2d_20/Conv2DConv2Dconv2d_19/BiasAdd:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
conv2d_20/Conv2Dª
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp§
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2
conv2d_20/BiasAdd³
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_21/Conv2D/ReadVariableOpÍ
conv2d_21/Conv2DConv2Dconv2d_20/BiasAdd:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	*
paddingVALID*
strides
2
conv2d_21/Conv2Dª
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp§
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:nn	2
conv2d_21/BiasAdd³
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_22/Conv2D/ReadVariableOpÍ
conv2d_22/Conv2DConv2Dconv2d_21/BiasAdd:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	*
paddingVALID*
strides
2
conv2d_22/Conv2Dª
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp§
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	2
conv2d_22/BiasAdd³
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_23/Conv2D/ReadVariableOpÍ
conv2d_23/Conv2DConv2Dconv2d_22/BiasAdd:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	*
paddingVALID*
strides
2
conv2d_23/Conv2Dª
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp§
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:bb	2
conv2d_23/BiasAdd³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_24/Conv2D/ReadVariableOpÍ
conv2d_24/Conv2DConv2Dconv2d_23/BiasAdd:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	*
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp§
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	2
conv2d_24/BiasAdd³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_25/Conv2D/ReadVariableOpÍ
conv2d_25/Conv2DConv2Dconv2d_24/BiasAdd:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	*
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp§
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:VV	2
conv2d_25/BiasAdd³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_26/Conv2D/ReadVariableOpÍ
conv2d_26/Conv2DConv2Dconv2d_25/BiasAdd:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp§
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	2
conv2d_26/BiasAdd
IdentityIdentityconv2d_26/BiasAdd:output:0!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_26_layer_call_and_return_conditional_losses_46806

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:PP	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:VV	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:VV	
 
_user_specified_nameinputs


)__inference_conv2d_21_layer_call_fn_47445

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:nn	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_467262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:nn	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs


ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_47493

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:\\	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:bb	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:bb	
 
_user_specified_nameinputs
à,
Ý
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_46813

inputs)
conv2d_19_46695:	
conv2d_19_46697:	)
conv2d_20_46711:		
conv2d_20_46713:	)
conv2d_21_46727:		
conv2d_21_46729:	)
conv2d_22_46743:		
conv2d_22_46745:	)
conv2d_23_46759:		
conv2d_23_46761:	)
conv2d_24_46775:		
conv2d_24_46777:	)
conv2d_25_46791:		
conv2d_25_46793:	)
conv2d_26_46807:		
conv2d_26_46809:	
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_46695conv2d_19_46697*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_466942#
!conv2d_19/StatefulPartitionedCall¹
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_46711conv2d_20_46713*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_467102#
!conv2d_20/StatefulPartitionedCall¹
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_46727conv2d_21_46729*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:nn	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_467262#
!conv2d_21/StatefulPartitionedCall¹
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_46743conv2d_22_46745*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_467422#
!conv2d_22/StatefulPartitionedCall¹
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_46759conv2d_23_46761*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:bb	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_467582#
!conv2d_23/StatefulPartitionedCall¹
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_46775conv2d_24_46777*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:\\	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_467742#
!conv2d_24/StatefulPartitionedCall¹
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_46791conv2d_25_46793*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:VV	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_467902#
!conv2d_25/StatefulPartitionedCall¹
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_46807conv2d_26_46809*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_468062#
!conv2d_26/StatefulPartitionedCall
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_47455

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:hh	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:nn	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:nn	
 
_user_specified_nameinputs


ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_46774

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:\\	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:\\	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:bb	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:bb	
 
_user_specified_nameinputs


ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_46742

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:hh	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:nn	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:nn	
 
_user_specified_nameinputs
Å
ß
3__inference_big_conv2d_stacked8_layer_call_fn_47351

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall¸
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
 *&
_output_shapes
:PP	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_468132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

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
ã,
Þ
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47171
input_1)
conv2d_19_47130:	
conv2d_19_47132:	)
conv2d_20_47135:		
conv2d_20_47137:	)
conv2d_21_47140:		
conv2d_21_47142:	)
conv2d_22_47145:		
conv2d_22_47147:	)
conv2d_23_47150:		
conv2d_23_47152:	)
conv2d_24_47155:		
conv2d_24_47157:	)
conv2d_25_47160:		
conv2d_25_47162:	)
conv2d_26_47165:		
conv2d_26_47167:	
identity¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_19_47130conv2d_19_47132*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_466942#
!conv2d_19/StatefulPartitionedCall¹
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_47135conv2d_20_47137*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_467102#
!conv2d_20/StatefulPartitionedCall¹
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_47140conv2d_21_47142*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:nn	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_467262#
!conv2d_21/StatefulPartitionedCall¹
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_47145conv2d_22_47147*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_467422#
!conv2d_22/StatefulPartitionedCall¹
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_47150conv2d_23_47152*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:bb	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_467582#
!conv2d_23/StatefulPartitionedCall¹
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_47155conv2d_24_47157*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:\\	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_467742#
!conv2d_24/StatefulPartitionedCall¹
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_25_47160conv2d_25_47162*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:VV	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_467902#
!conv2d_25/StatefulPartitionedCall¹
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_47165conv2d_26_47167*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:PP	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_468062#
!conv2d_26/StatefulPartitionedCall
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*
T0*&
_output_shapes
:PP	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
úF
 

!__inference__traced_restore_47669
file_prefix;
!assignvariableop_conv2d_19_kernel:	/
!assignvariableop_1_conv2d_19_bias:	=
#assignvariableop_2_conv2d_20_kernel:		/
!assignvariableop_3_conv2d_20_bias:	=
#assignvariableop_4_conv2d_21_kernel:		/
!assignvariableop_5_conv2d_21_bias:	=
#assignvariableop_6_conv2d_22_kernel:		/
!assignvariableop_7_conv2d_22_bias:	=
#assignvariableop_8_conv2d_23_kernel:		/
!assignvariableop_9_conv2d_23_bias:	>
$assignvariableop_10_conv2d_24_kernel:		0
"assignvariableop_11_conv2d_24_bias:	>
$assignvariableop_12_conv2d_25_kernel:		0
"assignvariableop_13_conv2d_25_bias:	>
$assignvariableop_14_conv2d_26_kernel:		0
"assignvariableop_15_conv2d_26_bias:	
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_19_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_19_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_20_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_20_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_23_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_23_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_24_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_24_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_25_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_25_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_26_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_26_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
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
_user_specified_namefile_prefix"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0<
	conv2d_26/
StatefulPartitionedCall:0PP	tensorflow/serving/predict:¹Å
½q
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
n_default_save_signature"¼m
_tf_keras_network m{"name": "big_conv2d_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "big_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_26", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "big_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_26", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ü


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"×	
_tf_keras_layer½	{"name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
þ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"Ù	
_tf_keras_layer¿	{"name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_19", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 122, 122, 9]}}
þ


kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*s&call_and_return_all_conditional_losses
t__call__"Ù	
_tf_keras_layer¿	{"name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_20", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 116, 116, 9]}}


!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*u&call_and_return_all_conditional_losses
v__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_21", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 110, 110, 9]}}


'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_22", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 104, 104, 9]}}
ÿ


-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"Ú	
_tf_keras_layerÀ	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_23", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 98, 98, 9]}}
ÿ


3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ú	
_tf_keras_layerÀ	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_24", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 92, 92, 9]}}
ÿ


9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*}&call_and_return_all_conditional_losses
~__call__"Ú	
_tf_keras_layerÀ	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_25", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 86, 86, 9]}}
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
Ê

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
*:(	2conv2d_19/kernel
:	2conv2d_19/bias
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
*:(		2conv2d_20/kernel
:	2conv2d_20/bias
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
*:(		2conv2d_21/kernel
:	2conv2d_21/bias
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
*:(		2conv2d_22/kernel
:	2conv2d_22/bias
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
*:(		2conv2d_23/kernel
:	2conv2d_23/bias
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
*:(		2conv2d_24/kernel
:	2conv2d_24/bias
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
*:(		2conv2d_25/kernel
:	2conv2d_25/bias
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
*:(		2conv2d_26/kernel
:	2conv2d_26/bias
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
2
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47262
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47314
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47127
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47171À
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
kwonlydefaultsª 
annotationsª *
 
2
3__inference_big_conv2d_stacked8_layer_call_fn_46848
3__inference_big_conv2d_stacked8_layer_call_fn_47351
3__inference_big_conv2d_stacked8_layer_call_fn_47388
3__inference_big_conv2d_stacked8_layer_call_fn_47083À
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
kwonlydefaultsª 
annotationsª *
 
ß2Ü
 __inference__wrapped_model_46677·
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
annotationsª *'¢$
"
input_1
î2ë
D__inference_conv2d_19_layer_call_and_return_conditional_losses_47398¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_19_layer_call_fn_47407¢
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
annotationsª *
 
î2ë
D__inference_conv2d_20_layer_call_and_return_conditional_losses_47417¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_20_layer_call_fn_47426¢
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
annotationsª *
 
î2ë
D__inference_conv2d_21_layer_call_and_return_conditional_losses_47436¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_21_layer_call_fn_47445¢
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
annotationsª *
 
î2ë
D__inference_conv2d_22_layer_call_and_return_conditional_losses_47455¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_22_layer_call_fn_47464¢
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
annotationsª *
 
î2ë
D__inference_conv2d_23_layer_call_and_return_conditional_losses_47474¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_23_layer_call_fn_47483¢
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
annotationsª *
 
î2ë
D__inference_conv2d_24_layer_call_and_return_conditional_losses_47493¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_24_layer_call_fn_47502¢
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
annotationsª *
 
î2ë
D__inference_conv2d_25_layer_call_and_return_conditional_losses_47512¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_25_layer_call_fn_47521¢
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
annotationsª *
 
î2ë
D__inference_conv2d_26_layer_call_and_return_conditional_losses_47531¢
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
annotationsª *
 
Ó2Ð
)__inference_conv2d_26_layer_call_fn_47540¢
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
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_47210input_1"
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
annotationsª *
 
 __inference__wrapped_model_46677{!"'(-.349:1¢.
'¢$
"
input_1
ª "4ª1
/
	conv2d_26"
	conv2d_26PP	Å
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47127s!"'(-.349:9¢6
/¢,
"
input_1
p 

 
ª "$¢!

0PP	
 Å
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47171s!"'(-.349:9¢6
/¢,
"
input_1
p

 
ª "$¢!

0PP	
 Ä
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47262r!"'(-.349:8¢5
.¢+
!
inputs
p 

 
ª "$¢!

0PP	
 Ä
N__inference_big_conv2d_stacked8_layer_call_and_return_conditional_losses_47314r!"'(-.349:8¢5
.¢+
!
inputs
p

 
ª "$¢!

0PP	
 
3__inference_big_conv2d_stacked8_layer_call_fn_46848f!"'(-.349:9¢6
/¢,
"
input_1
p 

 
ª "PP	
3__inference_big_conv2d_stacked8_layer_call_fn_47083f!"'(-.349:9¢6
/¢,
"
input_1
p

 
ª "PP	
3__inference_big_conv2d_stacked8_layer_call_fn_47351e!"'(-.349:8¢5
.¢+
!
inputs
p 

 
ª "PP	
3__inference_big_conv2d_stacked8_layer_call_fn_47388e!"'(-.349:8¢5
.¢+
!
inputs
p

 
ª "PP	¤
D__inference_conv2d_19_layer_call_and_return_conditional_losses_47398\0¢-
&¢#
!
inputs
ª "$¢!

0zz	
 |
)__inference_conv2d_19_layer_call_fn_47407O0¢-
&¢#
!
inputs
ª "zz	¢
D__inference_conv2d_20_layer_call_and_return_conditional_losses_47417Z.¢+
$¢!

inputszz	
ª "$¢!

0tt	
 z
)__inference_conv2d_20_layer_call_fn_47426M.¢+
$¢!

inputszz	
ª "tt	¢
D__inference_conv2d_21_layer_call_and_return_conditional_losses_47436Z.¢+
$¢!

inputstt	
ª "$¢!

0nn	
 z
)__inference_conv2d_21_layer_call_fn_47445M.¢+
$¢!

inputstt	
ª "nn	¢
D__inference_conv2d_22_layer_call_and_return_conditional_losses_47455Z!".¢+
$¢!

inputsnn	
ª "$¢!

0hh	
 z
)__inference_conv2d_22_layer_call_fn_47464M!".¢+
$¢!

inputsnn	
ª "hh	¢
D__inference_conv2d_23_layer_call_and_return_conditional_losses_47474Z'(.¢+
$¢!

inputshh	
ª "$¢!

0bb	
 z
)__inference_conv2d_23_layer_call_fn_47483M'(.¢+
$¢!

inputshh	
ª "bb	¢
D__inference_conv2d_24_layer_call_and_return_conditional_losses_47493Z-..¢+
$¢!

inputsbb	
ª "$¢!

0\\	
 z
)__inference_conv2d_24_layer_call_fn_47502M-..¢+
$¢!

inputsbb	
ª "\\	¢
D__inference_conv2d_25_layer_call_and_return_conditional_losses_47512Z34.¢+
$¢!

inputs\\	
ª "$¢!

0VV	
 z
)__inference_conv2d_25_layer_call_fn_47521M34.¢+
$¢!

inputs\\	
ª "VV	¢
D__inference_conv2d_26_layer_call_and_return_conditional_losses_47531Z9:.¢+
$¢!

inputsVV	
ª "$¢!

0PP	
 z
)__inference_conv2d_26_layer_call_fn_47540M9:.¢+
$¢!

inputsVV	
ª "PP	®
#__inference_signature_wrapper_47210!"'(-.349:<¢9
¢ 
2ª/
-
input_1"
input_1"4ª1
/
	conv2d_26"
	conv2d_26PP	