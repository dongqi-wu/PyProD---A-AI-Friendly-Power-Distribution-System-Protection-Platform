

# calculate the reward of the training agent
def step_reward(flags, action, env):
    R = 0
    # if no fault in the system (before fault time)
    if not flags['time']:
        if action: # trip before fault
            R = -150
        else:      # no trip
            R = 5

    # if after fault and outside region
    if flags['time'] and not flags['area']:
        if action:
            R = -150
        else:
            R = 5

    # if after fault, within region, this agent should trip
    if flags['time'] and flags['area'] and flags['coord'] and not flags['cleared']:
        if action:
            R = 100
        else:
            R = -10

    # if after fault, withing region, but need to wait for a neighbour
    if flags['time'] and flags['area'] and not flags['coord'] and not flags['cleared']:
        if action:
            R = -100
        else:
            R = 5


    # if fault has been cleared:
    if flags['time'] and flags['cleared']:
        if action:
            R = -100
        else:
            R = 2
    
    
    return R


# calculate the reward of the training agent
def time_increasing_reward(flags, action, env):
    R = 0
    timeDiff = floor(env.currStep - env.fault.T / env.ts)
    # if no fault in the system (before fault time)
    if not flags['time']:
        if action: # trip before fault
            R = -150
        else:      # no trip
            R = 5

    # if after fault and outside region
    if flags['time'] and not flags['area']:
        if action:
            R = -150
        else:
            R = 5

    # if after fault, within region, this agent should trip
    if flags['time'] and flags['area'] and flags['coord'] and not flags['cleared']:
        if action:
            R = 100
        else:
            R = -3 * timeDiff

    # if after fault, withing region, but need to wait for a neighbour
    if flags['time'] and flags['area'] and not flags['coord'] and not flags['cleared']:
        if action:
            R = -100
        else:
            R = 0


    # if fault has been cleared:
    if flags['time'] and flags['cleared']:
        if action:
            R = -100
        else:
            R = 2
    
    
    return R


        
