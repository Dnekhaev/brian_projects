from brian2 import ms

tau = 10*ms # tau for neuron's voltage 'v'
tau_I = 15*ms # tau for neuron's current 'I'
tau_h = 50*ms # tau for neuron's treshold
wmax = 1.
decay = 0.001

eqs_input_neuron = '''
rates : Hz
da/dt = -a/alpha : 1
dtheta/dt = -theta/beta : 1
diff = a - c_diff * theta : 1
train :1
'''

eqs_output_neuron = '''
dv/dt = (-v+I)/tau : 1 (unless refractory)
dI_inp/dt = -I_inp/tau_I :1
dI_intr/dt = -I_intr/tau_I :1
I_teacher :1
I = I_inp + I_intr + I_teacher : 1
da/dt = -a/alpha : 1
dtheta/dt = -theta/beta : 1
diff = a - c_diff * theta : 1
dhold_output/dt = -hold_output/tau_h  : 1
fixed_hold_output = clip(hold_output, 1, 5) :1
train :1
'''
#fixed_hold_output = clip(hold_output, 1, 100) :1

eqs_input_syn = '''
w : 1
'''

# equations that describe changes if presynaptic spike of the forward-riented synapse of input layer occures
eqs_input_pre = '''
I_inp_post += w * c_inp
a_pre += 1./classes *1*ms/(alpha)
theta_pre += 1./classes *1*ms/(beta)
'''

# equations that describe changes if postsynaptic spike of the forward synapse occures
eqs_input_post = '''
a_post += 1./n_input *1*ms/(alpha)
theta_post += 1./n_input *1*ms/(beta)
w = clip(w + train_pre*(-decay + lr*diff_pre), 0, wmax)
'''

eqs_intrinsic_output_syn = '''
w: 1
'''

# equations that describe changes if spike of the intr output synapse occures
eqs_intrinsic_output_pre = '''
I_intr_post += w * c_intr_out
'''

eqs_intrinsic_output_post = '''
w = clip(w + train_pre*(-decay-lr_intr*diff_pre), -wmax, 0)
'''


reset_output = '''
v = 0
hold_output += 0.1*classes
'''
