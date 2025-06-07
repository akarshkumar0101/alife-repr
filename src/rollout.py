import jax
import jax.numpy as jnp
from jax.random import split

def rollout_simulation(rng, params, s0=None,
                       substrate=None, fm=None, rollout_steps=256, time_sampling='final', img_size=224, return_state=False):
    """
    Rollout a simulation described by the specified substrate and parameters.

    Parameters
    ----------
    rng : jax rng seed for the rollout
    params : parameters to configure the simulation within the substrate
    s0 : (optional) initial state of the simulation. If None, then substrate.init_state(rng, params) is used.
    substrate : the substrate object
    fm : the foundation model object. If None, then no image embedding is calculated.
    rollout_steps : number of timesteps to run simulation for
    time_sampling : one of either
        - 'final': return only the final state data (default)
        - 'video': return the entire rollout
        - (K, chunk_ends): return the rollout at K sampled intervals, if chunk_ends is True then end of intervals is sampled
    img_size : image size to render at. Leave at 224 to avoid resizing again for CLIP.
    return_state : return the state data, leave as False, unless you really need it.

    Returns
    ----------
    A dictionary containing
    'state' : the state data of the simulation, None if return_state is False
        shape: (...)
    'rgb' : the image data of the simulation,
        shape (H, W, C)
    'z' : the image embedding of the simulation using the foundation model,
        shape (D)

    If time_sampling is 'video' then the returned shapes become (rollout_steps, ...).
    If time_sampling is an int then the returned shapes become (time_sampling, ...).

    ----------
    This function should be used like this:
    ```
    fm = create_foundation_model('clip')
    substrate = create_substrate('lenia')
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=256, time_sampling=8, img_size=224, return_state=False)
    rollout_fn = jax.jit(rollout_fn) # jit for speed
    # now you can use rollout_fn as you need...
    rng = jax.random.PRNGKey(0)
    params = substrate.default_params(rng)
    rollout_data = rollout_fn(rng, params)
    ```

    Note: 
    - when time_sampling is 'final', the function returns data for the T timestep.
    - when time_sampling is 'video', the function returns data for the [0, ..., T-1] timesteps.
    - when time_sampling is (K, False), the function returns data for the [0, T//K, T//K * 2, ..., T - T//K] timesteps.
    - when time_sampling is (K, True), the function returns data for the [T//K, T//K * 2, ..., T] timesteps.
    """

    if s0 is None:
        s0 = substrate.init_state(rng, params)
    embed_img_fn = (lambda img: None) if fm is None else fm.embed_img
    
    if time_sampling == 'final': # return only the final state
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        state_final, _ = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        img = substrate.render_state(state_final, params=params, img_size=img_size)
        z = embed_img_fn(img)
        return dict(rgb=img, z=z, state=(state_final if return_state else None))
    elif time_sampling == 'video': # return the entire rollout
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            img = substrate.render_state(state, params=params, img_size=img_size)
            z = embed_img_fn(img)
            return next_state, dict(rgb=img, z=z, state=(state if return_state else None))
        _, data = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        return data
    elif isinstance(time_sampling, int) or isinstance(time_sampling, tuple): # return the rollout at K sampled intervals
        K, chunk_ends = time_sampling if isinstance(time_sampling, tuple) else (time_sampling, False)
        chunk_steps = rollout_steps // K
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, state
        _, state_vid = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        if chunk_ends:
            idx_sample = jnp.arange(chunk_steps-1, rollout_steps, chunk_steps)
        else:
            idx_sample = jnp.arange(0, rollout_steps, chunk_steps)
        state_vid = jax.tree.map(lambda x: x[idx_sample], state_vid)
        def render_state(_, state):
            img = substrate.render_state(state, params=params, img_size=img_size)
            z = embed_img_fn(img)
            return _, dict(state=state, rgb=img, z=z)
        _, data = jax.lax.scan(render_state, None, state_vid)
        return data
    # elif isinstance(time_sampling, int) or isinstance(time_sampling, tuple): # return the rollout at K sampled intervals
    #     K, chunk_ends = time_sampling if isinstance(time_sampling, tuple) else (time_sampling, False)
    #     assert rollout_steps % K == 0
    #     chunk_steps = rollout_steps // K
    #     print(rollout_steps, K, chunk_steps)
    #     def step_fn(state, _rng):
    #         next_state = substrate.step_state(_rng, state, params)
    #         return next_state, None
    #     def chunk_fn(state, _rng):
    #         next_state, _ = jax.lax.scan(step_fn, state, split(_rng, chunk_steps))
    #         state_to_use = next_state if chunk_ends else state
    #         img = substrate.render_state(state_to_use, params=params, img_size=img_size)
    #         z = embed_img_fn(img)
    #         return next_state, dict(rgb=img, z=z, state=(state_to_use if return_state else None))
    #     state_final, data = jax.lax.scan(chunk_fn, s0, split(rng, K))
    #     return data
    else:
        raise ValueError(f"time_sampling {time_sampling} not recognized")