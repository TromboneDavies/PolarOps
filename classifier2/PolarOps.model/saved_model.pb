§ö

Ô
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68©	
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð`* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	Ð`*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:`*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:``*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:`*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:``*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:`*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:``*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:`*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:`*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð`*'
shared_nameAdam/dense_20/kernel/m

*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes
:	Ð`*
dtype0

Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:`*
dtype0

Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_21/kernel/m

*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes

:``*
dtype0

Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:`*
dtype0

Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_22/kernel/m

*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:``*
dtype0

Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:`*
dtype0

Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_23/kernel/m

*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:``*
dtype0

Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:`*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:`*
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0

Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð`*'
shared_nameAdam/dense_20/kernel/v

*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes
:	Ð`*
dtype0

Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:`*
dtype0

Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_21/kernel/v

*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:``*
dtype0

Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:`*
dtype0

Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_22/kernel/v

*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:``*
dtype0

Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:`*
dtype0

Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*'
shared_nameAdam/dense_23/kernel/v

*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:``*
dtype0

Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:`*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:`*
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
³V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*îU
valueäUBáU BÚU
Ð
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¥
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
¥
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses* 
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
¥
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6_random_generator
7__call__
*8&call_and_return_all_conditional_losses* 
¦

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
¥
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses* 
¦

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
¥
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses* 
¦

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses*

_iter

`beta_1

abeta_2
	bdecay
clearning_ratem§m¨*m©+mª9m«:m¬Hm­Im®Wm¯Xm°v±v²*v³+v´9vµ:v¶Hv·Iv¸Wv¹Xvº*
J
0
1
*2
+3
94
:5
H6
I7
W8
X9*
J
0
1
*2
+3
94
:5
H6
I7
W8
X9*
* 
°
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

iserving_default* 
* 
* 
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

W0
X1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
0
1
2
3
4
5
6
7
	8

9*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
 	variables
¡	keras_api*
M

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

 	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¢0
£1*

¥	variables*
|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

 serving_default_dropout_20_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿÐ
î
StatefulPartitionedCallStatefulPartitionedCall serving_default_dropout_20_inputdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_84630
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
__inference__traced_save_85005

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/v*3
Tin,
*2(*
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
!__inference__traced_restore_85132í
)
Ö
G__inference_sequential_4_layer_call_and_return_conditional_losses_84059

inputs!
dense_20_83957:	Ð`
dense_20_83959:` 
dense_21_83981:``
dense_21_83983:` 
dense_22_84005:``
dense_22_84007:` 
dense_23_84029:``
dense_23_84031:` 
dense_24_84053:`
dense_24_84055:
identity¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¼
dropout_20/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_83943
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_20_83957dense_20_83959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_83956Þ
dropout_21/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_83967
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_21_83981dense_21_83983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_83980Þ
dropout_22/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_83991
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_84005dense_22_84007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_84004Þ
dropout_23/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84015
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_23_84029dense_23_84031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_84028Þ
dropout_24/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84039
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dense_24_84053dense_24_84055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_84052x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_84015

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
®=
Ü	
 __inference__wrapped_model_83931
dropout_20_inputG
4sequential_4_dense_20_matmul_readvariableop_resource:	Ð`C
5sequential_4_dense_20_biasadd_readvariableop_resource:`F
4sequential_4_dense_21_matmul_readvariableop_resource:``C
5sequential_4_dense_21_biasadd_readvariableop_resource:`F
4sequential_4_dense_22_matmul_readvariableop_resource:``C
5sequential_4_dense_22_biasadd_readvariableop_resource:`F
4sequential_4_dense_23_matmul_readvariableop_resource:``C
5sequential_4_dense_23_biasadd_readvariableop_resource:`F
4sequential_4_dense_24_matmul_readvariableop_resource:`C
5sequential_4_dense_24_biasadd_readvariableop_resource:
identity¢,sequential_4/dense_20/BiasAdd/ReadVariableOp¢+sequential_4/dense_20/MatMul/ReadVariableOp¢,sequential_4/dense_21/BiasAdd/ReadVariableOp¢+sequential_4/dense_21/MatMul/ReadVariableOp¢,sequential_4/dense_22/BiasAdd/ReadVariableOp¢+sequential_4/dense_22/MatMul/ReadVariableOp¢,sequential_4/dense_23/BiasAdd/ReadVariableOp¢+sequential_4/dense_23/MatMul/ReadVariableOp¢,sequential_4/dense_24/BiasAdd/ReadVariableOp¢+sequential_4/dense_24/MatMul/ReadVariableOpq
 sequential_4/dropout_20/IdentityIdentitydropout_20_input*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ¡
+sequential_4/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_20_matmul_readvariableop_resource*
_output_shapes
:	Ð`*
dtype0¸
sequential_4/dense_20/MatMulMatMul)sequential_4/dropout_20/Identity:output:03sequential_4/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
,sequential_4/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_20_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¸
sequential_4/dense_20/BiasAddBiasAdd&sequential_4/dense_20/MatMul:product:04sequential_4/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`|
sequential_4/dense_20/ReluRelu&sequential_4/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 sequential_4/dropout_21/IdentityIdentity(sequential_4/dense_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
+sequential_4/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_21_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0¸
sequential_4/dense_21/MatMulMatMul)sequential_4/dropout_21/Identity:output:03sequential_4/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
,sequential_4/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_21_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¸
sequential_4/dense_21/BiasAddBiasAdd&sequential_4/dense_21/MatMul:product:04sequential_4/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`|
sequential_4/dense_21/ReluRelu&sequential_4/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 sequential_4/dropout_22/IdentityIdentity(sequential_4/dense_21/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
+sequential_4/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_22_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0¸
sequential_4/dense_22/MatMulMatMul)sequential_4/dropout_22/Identity:output:03sequential_4/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
,sequential_4/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_22_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¸
sequential_4/dense_22/BiasAddBiasAdd&sequential_4/dense_22/MatMul:product:04sequential_4/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`|
sequential_4/dense_22/ReluRelu&sequential_4/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 sequential_4/dropout_23/IdentityIdentity(sequential_4/dense_22/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
+sequential_4/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_23_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0¸
sequential_4/dense_23/MatMulMatMul)sequential_4/dropout_23/Identity:output:03sequential_4/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
,sequential_4/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_23_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¸
sequential_4/dense_23/BiasAddBiasAdd&sequential_4/dense_23/MatMul:product:04sequential_4/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`|
sequential_4/dense_23/ReluRelu&sequential_4/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 sequential_4/dropout_24/IdentityIdentity(sequential_4/dense_23/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
+sequential_4/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_24_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0¸
sequential_4/dense_24/MatMulMatMul)sequential_4/dropout_24/Identity:output:03sequential_4/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_4/dense_24/BiasAddBiasAdd&sequential_4/dense_24/MatMul:product:04sequential_4/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/dense_24/SigmoidSigmoid&sequential_4/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!sequential_4/dense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^sequential_4/dense_20/BiasAdd/ReadVariableOp,^sequential_4/dense_20/MatMul/ReadVariableOp-^sequential_4/dense_21/BiasAdd/ReadVariableOp,^sequential_4/dense_21/MatMul/ReadVariableOp-^sequential_4/dense_22/BiasAdd/ReadVariableOp,^sequential_4/dense_22/MatMul/ReadVariableOp-^sequential_4/dense_23/BiasAdd/ReadVariableOp,^sequential_4/dense_23/MatMul/ReadVariableOp-^sequential_4/dense_24/BiasAdd/ReadVariableOp,^sequential_4/dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2\
,sequential_4/dense_20/BiasAdd/ReadVariableOp,sequential_4/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_20/MatMul/ReadVariableOp+sequential_4/dense_20/MatMul/ReadVariableOp2\
,sequential_4/dense_21/BiasAdd/ReadVariableOp,sequential_4/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_21/MatMul/ReadVariableOp+sequential_4/dense_21/MatMul/ReadVariableOp2\
,sequential_4/dense_22/BiasAdd/ReadVariableOp,sequential_4/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_22/MatMul/ReadVariableOp+sequential_4/dense_22/MatMul/ReadVariableOp2\
,sequential_4/dense_23/BiasAdd/ReadVariableOp,sequential_4/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_23/MatMul/ReadVariableOp+sequential_4/dense_23/MatMul/ReadVariableOp2\
,sequential_4/dense_24/BiasAdd/ReadVariableOp,sequential_4/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_24/MatMul/ReadVariableOp+sequential_4/dense_24/MatMul/ReadVariableOp:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input


ô
C__inference_dense_21_layer_call_and_return_conditional_losses_84724

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_24_layer_call_and_return_conditional_losses_84865

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


d
E__inference_dropout_24_layer_call_and_return_conditional_losses_84112

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
¹

ü
,__inference_sequential_4_layer_call_fn_84356
dropout_20_input
unknown:	Ð`
	unknown_0:`
	unknown_1:``
	unknown_2:`
	unknown_3:``
	unknown_4:`
	unknown_5:``
	unknown_6:`
	unknown_7:`
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCalldropout_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_84308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input


d
E__inference_dropout_24_layer_call_and_return_conditional_losses_84845

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_84039

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
À

(__inference_dense_22_layer_call_fn_84760

inputs
unknown:``
	unknown_0:`
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_84004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_84786

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_22_layer_call_and_return_conditional_losses_84771

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_24_layer_call_and_return_conditional_losses_84052

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_84833

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
1
õ
G__inference_sequential_4_layer_call_and_return_conditional_losses_84524

inputs:
'dense_20_matmul_readvariableop_resource:	Ð`6
(dense_20_biasadd_readvariableop_resource:`9
'dense_21_matmul_readvariableop_resource:``6
(dense_21_biasadd_readvariableop_resource:`9
'dense_22_matmul_readvariableop_resource:``6
(dense_22_biasadd_readvariableop_resource:`9
'dense_23_matmul_readvariableop_resource:``6
(dense_23_biasadd_readvariableop_resource:`9
'dense_24_matmul_readvariableop_resource:`6
(dense_24_biasadd_readvariableop_resource:
identity¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢dense_22/MatMul/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢dense_23/MatMul/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOpZ
dropout_20/IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	Ð`*
dtype0
dense_20/MatMulMatMuldropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`n
dropout_21/IdentityIdentitydense_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_21/MatMulMatMuldropout_21/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`n
dropout_22/IdentityIdentitydense_21/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`n
dropout_23/IdentityIdentitydense_22/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_23/MatMulMatMuldropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`n
dropout_24/IdentityIdentitydense_23/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0
dense_24/MatMulMatMuldropout_24/Identity:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_24_layer_call_fn_84828

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_22_layer_call_and_return_conditional_losses_84004

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_23_layer_call_and_return_conditional_losses_84818

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
õ
c
*__inference_dropout_20_layer_call_fn_84640

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_84244p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


d
E__inference_dropout_22_layer_call_and_return_conditional_losses_84751

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_83943

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


ò
,__inference_sequential_4_layer_call_fn_84480

inputs
unknown:	Ð`
	unknown_0:`
	unknown_1:``
	unknown_2:`
	unknown_3:``
	unknown_4:`
	unknown_5:``
	unknown_6:`
	unknown_7:`
	unknown_8:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_84308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


ò
,__inference_sequential_4_layer_call_fn_84455

inputs
unknown:	Ð`
	unknown_0:`
	unknown_1:``
	unknown_2:`
	unknown_3:``
	unknown_4:`
	unknown_5:``
	unknown_6:`
	unknown_7:`
	unknown_8:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_84059o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_84739

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

F
*__inference_dropout_23_layer_call_fn_84776

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84015`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_21_layer_call_fn_84687

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_84211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ã

(__inference_dense_20_layer_call_fn_84666

inputs
unknown:	Ð`
	unknown_0:`
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_83956o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
£)
à
G__inference_sequential_4_layer_call_and_return_conditional_losses_84390
dropout_20_input!
dense_20_84360:	Ð`
dense_20_84362:` 
dense_21_84366:``
dense_21_84368:` 
dense_22_84372:``
dense_22_84374:` 
dense_23_84378:``
dense_23_84380:` 
dense_24_84384:`
dense_24_84386:
identity¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCallÆ
dropout_20/PartitionedCallPartitionedCalldropout_20_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_83943
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_20_84360dense_20_84362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_83956Þ
dropout_21/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_83967
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_21_84366dense_21_84368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_83980Þ
dropout_22/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_83991
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_84372dense_22_84374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_84004Þ
dropout_23/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84015
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_23_84378dense_23_84380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_84028Þ
dropout_24/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84039
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dense_24_84384dense_24_84386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_84052x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input
À

(__inference_dense_24_layer_call_fn_84854

inputs
unknown:`
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_84052o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
¹

ü
,__inference_sequential_4_layer_call_fn_84082
dropout_20_input
unknown:	Ð`
	unknown_0:`
	unknown_1:``
	unknown_2:`
	unknown_3:``
	unknown_4:`
	unknown_5:``
	unknown_6:`
	unknown_7:`
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCalldropout_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_84059o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input


d
E__inference_dropout_20_layer_call_and_return_conditional_losses_84244

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


d
E__inference_dropout_23_layer_call_and_return_conditional_losses_84145

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
X
õ
G__inference_sequential_4_layer_call_and_return_conditional_losses_84603

inputs:
'dense_20_matmul_readvariableop_resource:	Ð`6
(dense_20_biasadd_readvariableop_resource:`9
'dense_21_matmul_readvariableop_resource:``6
(dense_21_biasadd_readvariableop_resource:`9
'dense_22_matmul_readvariableop_resource:``6
(dense_22_biasadd_readvariableop_resource:`9
'dense_23_matmul_readvariableop_resource:``6
(dense_23_biasadd_readvariableop_resource:`9
'dense_24_matmul_readvariableop_resource:`6
(dense_24_biasadd_readvariableop_resource:
identity¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢dense_22/MatMul/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢dense_23/MatMul/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOp]
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?{
dropout_20/dropout/MulMulinputs!dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐN
dropout_20/dropout/ShapeShapeinputs*
T0*
_output_shapes
:°
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ*
dtype0*
seed¹`f
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>È
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	Ð`*
dtype0
dense_20/MatMulMatMuldropout_20/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`]
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_21/dropout/MulMuldense_20/Relu:activations:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`c
dropout_21/dropout/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
:¼
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`*
seed2f
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_21/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`]
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_22/dropout/MulMuldense_21/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`c
dropout_22/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:¼
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`*
seed2f
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`]
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_23/dropout/MulMuldense_22/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`c
dropout_23/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:¼
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`*
seed2f
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:``*
dtype0
dense_23/MatMulMatMuldropout_23/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`]
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_24/dropout/MulMuldense_23/Relu:activations:0!dropout_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`c
dropout_24/dropout/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:¼
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`*
seed2f
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0
dense_24/MatMulMatMuldropout_24/dropout/Mul_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


d
E__inference_dropout_20_layer_call_and_return_conditional_losses_84657

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


ó
#__inference_signature_wrapper_84630
dropout_20_input
unknown:	Ð`
	unknown_0:`
	unknown_1:``
	unknown_2:`
	unknown_3:``
	unknown_4:`
	unknown_5:``
	unknown_6:`
	unknown_7:`
	unknown_8:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalldropout_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_83931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input


õ
C__inference_dense_20_layer_call_and_return_conditional_losses_84677

inputs1
matmul_readvariableop_resource:	Ð`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
¯P
õ
__inference__traced_save_85005
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: é
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH½
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Å
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: :	Ð`:`:``:`:``:`:``:`:`:: : : : : : : : : :	Ð`:`:``:`:``:`:``:`:`::	Ð`:`:``:`:``:`:``:`:`:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Ð`: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$	 

_output_shapes

:`: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Ð`: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$ 

_output_shapes

:``: 

_output_shapes
:`:$ 

_output_shapes

:`: 

_output_shapes
::%!

_output_shapes
:	Ð`: 

_output_shapes
:`:$  

_output_shapes

:``: !

_output_shapes
:`:$" 

_output_shapes

:``: #

_output_shapes
:`:$$ 

_output_shapes

:``: %

_output_shapes
:`:$& 

_output_shapes

:`: '

_output_shapes
::(

_output_shapes
: 
Ø
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_83967

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
À

(__inference_dense_21_layer_call_fn_84713

inputs
unknown:``
	unknown_0:`
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_83980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
1

G__inference_sequential_4_layer_call_and_return_conditional_losses_84424
dropout_20_input!
dense_20_84394:	Ð`
dense_20_84396:` 
dense_21_84400:``
dense_21_84402:` 
dense_22_84406:``
dense_22_84408:` 
dense_23_84412:``
dense_23_84414:` 
dense_24_84418:`
dense_24_84420:
identity¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢"dropout_20/StatefulPartitionedCall¢"dropout_21/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCallÖ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCalldropout_20_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_84244
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_20_84394dense_20_84396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_83956
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_84211
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_21_84400dense_21_84402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_83980
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_84178
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_84406dense_22_84408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_84004
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84145
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_23_84412dense_23_84414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_84028
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_23/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84112
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dense_24_84418dense_24_84420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_84052x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall:Z V
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
*
_user_specified_namedropout_20_input
Ü
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_84645

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
É
¤
!__inference__traced_restore_85132
file_prefix3
 assignvariableop_dense_20_kernel:	Ð`.
 assignvariableop_1_dense_20_bias:`4
"assignvariableop_2_dense_21_kernel:``.
 assignvariableop_3_dense_21_bias:`4
"assignvariableop_4_dense_22_kernel:``.
 assignvariableop_5_dense_22_bias:`4
"assignvariableop_6_dense_23_kernel:``.
 assignvariableop_7_dense_23_bias:`4
"assignvariableop_8_dense_24_kernel:`.
 assignvariableop_9_dense_24_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: =
*assignvariableop_19_adam_dense_20_kernel_m:	Ð`6
(assignvariableop_20_adam_dense_20_bias_m:`<
*assignvariableop_21_adam_dense_21_kernel_m:``6
(assignvariableop_22_adam_dense_21_bias_m:`<
*assignvariableop_23_adam_dense_22_kernel_m:``6
(assignvariableop_24_adam_dense_22_bias_m:`<
*assignvariableop_25_adam_dense_23_kernel_m:``6
(assignvariableop_26_adam_dense_23_bias_m:`<
*assignvariableop_27_adam_dense_24_kernel_m:`6
(assignvariableop_28_adam_dense_24_bias_m:=
*assignvariableop_29_adam_dense_20_kernel_v:	Ð`6
(assignvariableop_30_adam_dense_20_bias_v:`<
*assignvariableop_31_adam_dense_21_kernel_v:``6
(assignvariableop_32_adam_dense_21_bias_v:`<
*assignvariableop_33_adam_dense_22_kernel_v:``6
(assignvariableop_34_adam_dense_22_bias_v:`<
*assignvariableop_35_adam_dense_23_kernel_v:``6
(assignvariableop_36_adam_dense_23_bias_v:`<
*assignvariableop_37_adam_dense_24_kernel_v:`6
(assignvariableop_38_adam_dense_24_bias_v:
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ì
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_24_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_24_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_20_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_20_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_21_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_21_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_22_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_22_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_23_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_23_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_24_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_24_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_20_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_20_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_21_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_21_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_22_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_22_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_23_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_23_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_24_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_24_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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


d
E__inference_dropout_23_layer_call_and_return_conditional_losses_84798

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
õ0

G__inference_sequential_4_layer_call_and_return_conditional_losses_84308

inputs!
dense_20_84278:	Ð`
dense_20_84280:` 
dense_21_84284:``
dense_21_84286:` 
dense_22_84290:``
dense_22_84292:` 
dense_23_84296:``
dense_23_84298:` 
dense_24_84302:`
dense_24_84304:
identity¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢"dropout_20/StatefulPartitionedCall¢"dropout_21/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCallÌ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_84244
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_20_84278dense_20_84280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_83956
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_84211
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_21_84284dense_21_84286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_83980
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_84178
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_84290dense_22_84292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_84004
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84145
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_23_84296dense_23_84298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_84028
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_23/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84112
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dense_24_84302dense_24_84304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_84052x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÐ: : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_84692

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_23_layer_call_and_return_conditional_losses_84028

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


d
E__inference_dropout_21_layer_call_and_return_conditional_losses_84211

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


ô
C__inference_dense_21_layer_call_and_return_conditional_losses_83980

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

F
*__inference_dropout_22_layer_call_fn_84729

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_83991`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_23_layer_call_fn_84781

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_84145o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
À

(__inference_dense_23_layer_call_fn_84807

inputs
unknown:``
	unknown_0:`
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_84028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


õ
C__inference_dense_20_layer_call_and_return_conditional_losses_83956

inputs1
matmul_readvariableop_resource:	Ð`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
£
F
*__inference_dropout_20_layer_call_fn_84635

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_83943a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs

F
*__inference_dropout_21_layer_call_fn_84682

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_83967`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_22_layer_call_fn_84734

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_84178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

F
*__inference_dropout_24_layer_call_fn_84823

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_84039`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_83991

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


d
E__inference_dropout_21_layer_call_and_return_conditional_losses_84704

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs


d
E__inference_dropout_22_layer_call_and_return_conditional_losses_84178

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype0*
seed¹`[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
N
dropout_20_input:
"serving_default_dropout_20_input:0ÿÿÿÿÿÿÿÿÿÐ<
dense_240
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¼
ê
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
¼
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6_random_generator
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
»

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer

_iter

`beta_1

abeta_2
	bdecay
clearning_ratem§m¨*m©+mª9m«:m¬Hm­Im®Wm¯Xm°v±v²*v³+v´9vµ:v¶Hv·Iv¸Wv¹Xvº"
	optimizer
f
0
1
*2
+3
94
:5
H6
I7
W8
X9"
trackable_list_wrapper
f
0
1
*2
+3
94
:5
H6
I7
W8
X9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_4_layer_call_fn_84082
,__inference_sequential_4_layer_call_fn_84455
,__inference_sequential_4_layer_call_fn_84480
,__inference_sequential_4_layer_call_fn_84356À
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
ê2ç
G__inference_sequential_4_layer_call_and_return_conditional_losses_84524
G__inference_sequential_4_layer_call_and_return_conditional_losses_84603
G__inference_sequential_4_layer_call_and_return_conditional_losses_84390
G__inference_sequential_4_layer_call_and_return_conditional_losses_84424À
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
ÔBÑ
 __inference__wrapped_model_83931dropout_20_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_20_layer_call_fn_84635
*__inference_dropout_20_layer_call_fn_84640´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_20_layer_call_and_return_conditional_losses_84645
E__inference_dropout_20_layer_call_and_return_conditional_losses_84657´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
": 	Ð`2dense_20/kernel
:`2dense_20/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_20_layer_call_fn_84666¢
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
í2ê
C__inference_dense_20_layer_call_and_return_conditional_losses_84677¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_21_layer_call_fn_84682
*__inference_dropout_21_layer_call_fn_84687´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_21_layer_call_and_return_conditional_losses_84692
E__inference_dropout_21_layer_call_and_return_conditional_losses_84704´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:``2dense_21/kernel
:`2dense_21/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_21_layer_call_fn_84713¢
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
í2ê
C__inference_dense_21_layer_call_and_return_conditional_losses_84724¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_22_layer_call_fn_84729
*__inference_dropout_22_layer_call_fn_84734´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_22_layer_call_and_return_conditional_losses_84739
E__inference_dropout_22_layer_call_and_return_conditional_losses_84751´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:``2dense_22/kernel
:`2dense_22/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_22_layer_call_fn_84760¢
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
í2ê
C__inference_dense_22_layer_call_and_return_conditional_losses_84771¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_23_layer_call_fn_84776
*__inference_dropout_23_layer_call_fn_84781´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_23_layer_call_and_return_conditional_losses_84786
E__inference_dropout_23_layer_call_and_return_conditional_losses_84798´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:``2dense_23/kernel
:`2dense_23/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_23_layer_call_fn_84807¢
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
í2ê
C__inference_dense_23_layer_call_and_return_conditional_losses_84818¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_24_layer_call_fn_84823
*__inference_dropout_24_layer_call_fn_84828´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_24_layer_call_and_return_conditional_losses_84833
E__inference_dropout_24_layer_call_and_return_conditional_losses_84845´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:`2dense_24/kernel
:2dense_24/bias
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_24_layer_call_fn_84854¢
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
í2ê
C__inference_dense_24_layer_call_and_return_conditional_losses_84865¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
#__inference_signature_wrapper_84630dropout_20_input"
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
 
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
R

total

count
 	variables
¡	keras_api"
_tf_keras_metric
c

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
.
¥	variables"
_generic_user_object
':%	Ð`2Adam/dense_20/kernel/m
 :`2Adam/dense_20/bias/m
&:$``2Adam/dense_21/kernel/m
 :`2Adam/dense_21/bias/m
&:$``2Adam/dense_22/kernel/m
 :`2Adam/dense_22/bias/m
&:$``2Adam/dense_23/kernel/m
 :`2Adam/dense_23/bias/m
&:$`2Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
':%	Ð`2Adam/dense_20/kernel/v
 :`2Adam/dense_20/bias/v
&:$``2Adam/dense_21/kernel/v
 :`2Adam/dense_21/bias/v
&:$``2Adam/dense_22/kernel/v
 :`2Adam/dense_22/bias/v
&:$``2Adam/dense_23/kernel/v
 :`2Adam/dense_23/bias/v
&:$`2Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v¡
 __inference__wrapped_model_83931}
*+9:HIWX:¢7
0¢-
+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ
ª "3ª0
.
dense_24"
dense_24ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_20_layer_call_and_return_conditional_losses_84677]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÐ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 |
(__inference_dense_20_layer_call_fn_84666P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÐ
ª "ÿÿÿÿÿÿÿÿÿ`£
C__inference_dense_21_layer_call_and_return_conditional_losses_84724\*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 {
(__inference_dense_21_layer_call_fn_84713O*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "ÿÿÿÿÿÿÿÿÿ`£
C__inference_dense_22_layer_call_and_return_conditional_losses_84771\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 {
(__inference_dense_22_layer_call_fn_84760O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "ÿÿÿÿÿÿÿÿÿ`£
C__inference_dense_23_layer_call_and_return_conditional_losses_84818\HI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 {
(__inference_dense_23_layer_call_fn_84807OHI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "ÿÿÿÿÿÿÿÿÿ`£
C__inference_dense_24_layer_call_and_return_conditional_losses_84865\WX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_24_layer_call_fn_84854OWX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_20_layer_call_and_return_conditional_losses_84645^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÐ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÐ
 §
E__inference_dropout_20_layer_call_and_return_conditional_losses_84657^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÐ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÐ
 
*__inference_dropout_20_layer_call_fn_84635Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÐ
p 
ª "ÿÿÿÿÿÿÿÿÿÐ
*__inference_dropout_20_layer_call_fn_84640Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÐ
p
ª "ÿÿÿÿÿÿÿÿÿÐ¥
E__inference_dropout_21_layer_call_and_return_conditional_losses_84692\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 ¥
E__inference_dropout_21_layer_call_and_return_conditional_losses_84704\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 }
*__inference_dropout_21_layer_call_fn_84682O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "ÿÿÿÿÿÿÿÿÿ`}
*__inference_dropout_21_layer_call_fn_84687O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "ÿÿÿÿÿÿÿÿÿ`¥
E__inference_dropout_22_layer_call_and_return_conditional_losses_84739\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 ¥
E__inference_dropout_22_layer_call_and_return_conditional_losses_84751\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 }
*__inference_dropout_22_layer_call_fn_84729O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "ÿÿÿÿÿÿÿÿÿ`}
*__inference_dropout_22_layer_call_fn_84734O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "ÿÿÿÿÿÿÿÿÿ`¥
E__inference_dropout_23_layer_call_and_return_conditional_losses_84786\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 ¥
E__inference_dropout_23_layer_call_and_return_conditional_losses_84798\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 }
*__inference_dropout_23_layer_call_fn_84776O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "ÿÿÿÿÿÿÿÿÿ`}
*__inference_dropout_23_layer_call_fn_84781O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "ÿÿÿÿÿÿÿÿÿ`¥
E__inference_dropout_24_layer_call_and_return_conditional_losses_84833\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 ¥
E__inference_dropout_24_layer_call_and_return_conditional_losses_84845\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 }
*__inference_dropout_24_layer_call_fn_84823O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "ÿÿÿÿÿÿÿÿÿ`}
*__inference_dropout_24_layer_call_fn_84828O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "ÿÿÿÿÿÿÿÿÿ`Â
G__inference_sequential_4_layer_call_and_return_conditional_losses_84390w
*+9:HIWXB¢?
8¢5
+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
G__inference_sequential_4_layer_call_and_return_conditional_losses_84424w
*+9:HIWXB¢?
8¢5
+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
G__inference_sequential_4_layer_call_and_return_conditional_losses_84524m
*+9:HIWX8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÐ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
G__inference_sequential_4_layer_call_and_return_conditional_losses_84603m
*+9:HIWX8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÐ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_4_layer_call_fn_84082j
*+9:HIWXB¢?
8¢5
+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_84356j
*+9:HIWXB¢?
8¢5
+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_84455`
*+9:HIWX8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÐ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_84480`
*+9:HIWX8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÐ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
#__inference_signature_wrapper_84630
*+9:HIWXN¢K
¢ 
DªA
?
dropout_20_input+(
dropout_20_inputÿÿÿÿÿÿÿÿÿÐ"3ª0
.
dense_24"
dense_24ÿÿÿÿÿÿÿÿÿ