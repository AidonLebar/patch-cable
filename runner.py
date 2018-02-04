#!/usr/bin/env python3

import math
import numpy as np
import time
import random
import multiprocessing
import pygraphviz as pgv
import sys
import re

from input_buttons.input_reader import input_monitor


SAMPLE_RATE = 19200.0
VOLUME = 0.5
FRAME_SIZE = 1024

BEAT_32ND = SAMPLE_RATE / 32.0
BEAT_16TH = SAMPLE_RATE / 16.0
BEAT_8TH = SAMPLE_RATE / 8.0
BEAT_4TH = SAMPLE_RATE / 4.0
BEAT_HALF = SAMPLE_RATE / 2.0
BEAT_WHOLE = SAMPLE_RATE

BUTTON_1 = 1
BUTTON_2 = 2
BUTTON_3 = 3
BUTTON_4 = 4
BUTTON_5 = 5
BUTTON_6 = 6
BUTTON_7 = 7
POTENTIOMETER = 8

TWO_PI = 2.0 * math.pi


global_watchers = []

quit_threads = multiprocessing.Value('i', 0)

manager = multiprocessing.Manager()
shared_list = manager.list(([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


class Parameter:
    PARAM_CONSTANT = 'PARAM_CONSTANT'
    PARAM_CHAIN = 'PARAM_CHAIN'
    PARAM_INPUT = 'PARAM_INPUT'

    def __init__(self, param_value):
        self.param_value = param_value

        if type(param_value).__name__ == 'Chain':
            self.param_type = Parameter.PARAM_CHAIN
            self.cached_value = self.param_value.value
        elif isinstance(param_value, float):
            self.param_type = Parameter.PARAM_CONSTANT
            self.cached_value = self.param_value
        else:
            self.param_type = Parameter.PARAM_INPUT
            self.cached_value = shared_list[self.param_value - 1]
            global_watchers.append(self)

    @property
    def value(self):
        if self.param_type == Parameter.PARAM_CONSTANT:
            return self.param_value
        elif self.param_type == Parameter.PARAM_CHAIN:
            return self.param_value.value
        elif self.param_type == Parameter.PARAM_INPUT:
            return shared_list[self.param_value - 1]
        else:
            return 0

    def tick(self):
        if self.param_type == Parameter.PARAM_CHAIN:
            self.cached_value = self.param_value.value
        elif self.param_type == Parameter.PARAM_INPUT:
            self.cached_value = shared_list[self.param_value - 1]


class Node:
    def id(self, x):
        return x

    def __init__(self):
        self.upstream = []
        self.upstream_count = 0  # Caching
        self.downstream = []
        self.function = self.id
        self.value = self.function(0)

    def register_upstream(self, up):
        self.upstream.append(up)
        self.upstream_count = len(self.upstream)
        up.attach_downstream(self)
        return self

    def unregister_upstream(self, up):
        if up in self.upstream:
            self.upstream.remove(up)
            self.upstream_count = len(self.upstream)
            up.remove_downstream(self)
        return self

    def attach_downstream(self, ds):
        self.downstream.append(ds)

    def remove_downstream(self, ds):
        self.downstream.remove(ds)

    def register_downstream(self, ds):
        self.attach_downstream(ds)
        ds.register_upstream(self)
        return self

    def step(self):
        self.value = self.function(sum([up.value for up in self.upstream]) / self.upstream_count)

    def run_chain(self):
        self.step()
        for ds in self.downstream:
            ds.run_chain()

    def reset_chain(self):
        self.value = self.function(0)
        for ds in self.downstream:
            ds.reset_chain()

    def get_display_properties(self):
        return ''


class SourceNode(Node):
    def __init__(self):
        super().__init__()
        self._x = 0

    def step(self):
        self._x += 1
        self.value = self.function(self._x)

    def reset_chain(self):
        self._x = 0
        super().reset_chain()


class ChainStartNode(SourceNode):
    def __init__(self, start_param, gate=0.01):
        global global_watchers

        super().__init__()
        self.start_param = start_param
        self.gate = gate
        self.started = False
        self.chain = None

        global_watchers.append(self)

    def tick(self):
        if self.chain is not None:
            if self.start_param.value >= self.gate and not self.chain.started:
                self.chain.play_chain()
            elif self.start_param.value < self.gate and self.chain.started and not self.chain.terminating:
                self.chain.stop_chain()

    def reset_chain(self):
        self.started = False
        super().reset_chain()

    def get_display_properties(self):
        return 'Gate: {}'.format(self.gate)


class ChainTerminationNode(Node):
    def __init__(self, release_chain=None):
        super().__init__()
        self.release_chain = release_chain


class RandomNoiseNode(SourceNode):
    def noise_fn(self, _):
        return self.translate - 1.0 + random.random() * self.amplitude * 2.0

    def __init__(self, translate=0.0, amplitude=1.0):
        super().__init__()
        self.translate = translate
        self.amplitude = amplitude
        self.function = self.noise_fn

    def get_display_properties(self):
        return 'Translate: {}\nAmplitude: {}'.format(self.translate, self.amplitude)


class SineNode(SourceNode):
    def sin(self, x):
        return self.translate + self.amplitude * math.sin(TWO_PI * (x / SAMPLE_RATE)
                                                          * (self.frequency_offset + self.frequency.cached_value
                                                             * self.frequency_mulitplier))

    def __init__(
            self,
            frequency=Parameter(440.0),
            translate=0.0,
            amplitude=1.0,
            frequency_multiplier=1.0,
            frequency_offset=0.0
    ):
        super().__init__()
        self.frequency = frequency
        self.function = self.sin
        self.translate = translate
        self.amplitude = amplitude
        self.frequency_mulitplier = frequency_multiplier
        self.frequency_offset = frequency_offset

    def get_display_properties(self):
        return 'Translate: {}\nAmplitude: {}\nFrequency: {} + {} * {}'.format(
            self.translate,
            self.amplitude,
            self.frequency_offset,
            self.frequency.cached_value,
            self.frequency_mulitplier
        )


class SquareNode(SourceNode):
    def square(self, x):
        return math.copysign(1, math.sin(TWO_PI * (x / SAMPLE_RATE) * self.frequency.value))

    def __init__(self, frequency=Parameter(440.0)):
        super().__init__()
        self.frequency = frequency
        self.function = self.square

    def get_display_properties(self):
        return 'Frequency: {}'.format(
            self.frequency.cached_value
        )


class TriangleNode(SourceNode):
    def triangle(self, x):
        return self.translate + self.amplitude * (2 / math.pi)\
               * math.asin(math.sin(TWO_PI * (x / SAMPLE_RATE) * self.frequency.value))

    def __init__(self, frequency=Parameter(440.0), amplitude=1.0, translate=0.0):
        super().__init__()
        self.frequency = frequency
        self.function = self.triangle
        self.translate = translate
        self.amplitude = amplitude

    def get_display_properties(self):
        return 'Translate: {}\nAmplitude: {}\nFrequency: {}'.format(
            self.translate,
            self.amplitude,
            self.frequency.cached_value
        )


class KickDrumNode(SourceNode):  # kick drum
    def kick_drum(self, x):
        if 0 < x < self.length:
            return self.translate + random.random() * self.amplitude
        elif self.length <= x < self.length + self.sustain:
            return self.amplitude * math.sin(TWO_PI * (x / SAMPLE_RATE) * self.frequency.value)
        else:
            return 0

    def __init__(
            self,

            frequency=Parameter(75.0),
            length=0.005*SAMPLE_RATE,

            amplitude=2.0,
            translate=0.0,

            sustain=0.03*SAMPLE_RATE
    ):
        super().__init__()
        self.length = length
        self.frequency = frequency
        self.amplitude = amplitude
        self.function = self.kick_drum
        self.translate = translate
        self.sustain = sustain


class HiHatNode(SourceNode):  # hi-hat
    def hi_hat(self, x):
        if x < self.length:
            return self.translate + random.uniform(self.pass_filter, 1.0) * self.amplitude
        else:
            return 0

    def __init__(
            self,
            pass_filter=0.0,
            length=0.02*SAMPLE_RATE,
            amplitude=1.0,
            translate=0.0
    ):
        super().__init__()
        self.length = length
        self.pass_filter = pass_filter
        self.amplitude = amplitude
        self.function = self.hi_hat
        self.translate = translate


class SawtoothNode(SourceNode):
    def sawtooth(self, x):
        try:
            evaluation = self.amplitude * (-2.0 / math.pi *
                                           math.atan(1.0/math.tan(self.phase + x * math.pi / (SAMPLE_RATE / self.frequency.value))))
        except ZeroDivisionError:
            evaluation = 0
        return evaluation

    def __init__(self, frequency=Parameter(440.0), amplitude=1.0, phase=0.0):
        super().__init__()
        self.frequency = frequency
        self.function = self.sawtooth
        self.amplitude = amplitude
        self.phase = phase

    def get_display_properties(self):
        return 'Amplitude: {}\nFrequency: {}\nPhase: {}'.format(
            self.amplitude,
            self.frequency.cached_value,
            self.phase
        )


class BeatNode(SourceNode):
    def beat_fn(self, x):
        return self.translate + (self.amplitude if x % self.period_length <= self.beat_length else 0)

    def __init__(
            self,

            translate=0.0,
            amplitude=1.0,

            beat_length=BEAT_4TH,
            gap_length=BEAT_4TH
    ):
        super().__init__()
        self.translate = translate
        self.amplitude = amplitude
        self.beat_length = beat_length
        self.gap_length = gap_length
        self.period_length = self.beat_length + self.gap_length
        self.function = self.beat_fn


class FilterNode(Node):
    def filter_fn(self, x):
        return self.offset + (x * self.filter_param.value * self.multiplier)

    def __init__(self, filter_param, offset=0.0, multiplier=1.0):
        super().__init__()
        self.filter_param = filter_param
        self.offset = offset
        self.multiplier = multiplier

        self.function = self.filter_fn


class LinearAttackNode(Node):
    def attack_fn(self, y):
        return (1 - (max(self.duration - self._x, 0) / self.duration)) * y

    def __init__(self, duration=BEAT_HALF):
        super().__init__()
        self.duration = duration
        self.function = self.attack_fn
        self._x = 0

    def step(self):
        self._x += 1
        super().step()

    def reset_chain(self):
        self._x = 0
        super().reset_chain()

    def get_display_properties(self):
        return 'Duration: {} s'.format(
            self.duration / SAMPLE_RATE
        )


class LinearDecayNode(Node):
    def decay_fn(self, y):
        return (max(self.duration - self._x, 0) / self.duration) * y

    def __init__(self, duration=BEAT_HALF):
        super().__init__()
        self.duration = duration
        self.function = self.decay_fn
        self._x = 0

    def step(self):
        self._x += 1
        super().step()

    def reset_chain(self):
        self._x = 0
        super().reset_chain()

    def get_display_properties(self):
        return 'Duration: {} s'.format(
            self.duration / SAMPLE_RATE
        )


cached_repeat = range(FRAME_SIZE)


class Chain:
    def __init__(self, source_node, termination_node, duration=-1.0):
        global global_watchers

        global_watchers.append(self)

        self.source_node = source_node
        self.termination_node = termination_node
        self.old_termination_node = termination_node

        self.source_node.chain = self
        self.termination_node.chain = self

        self.stream = None
        self.started = False
        self.terminating = False

        self.time_elapsed = 0.0
        self.duration = duration
        self.old_duration = duration

        self.values = None

    def play_chain(self, save_values=False):
        if self.started:
            return

        import pyaudio
        p = pyaudio.PyAudio()

        sn = self.source_node

        if save_values:
            self.values = []

        def run_chain(nodes):
            while len(nodes) > 0:
                new_nodes = []
                for n in nodes:
                    n.step()
                    new_nodes.extend(n.downstream)
                nodes = list(set(new_nodes))

        def callback(_in_data, _frame_count, _time_info, _status):
            output_samples = []

            for _ in cached_repeat:
                run_chain([sn])
                output_samples.append(self.termination_node.value)

            return np.array(output_samples, dtype=np.float32) * VOLUME, pyaudio.paContinue

        if save_values:
            for i in range(int(self.duration)):
                self.time_elapsed += 1
                run_chain([sn])
                self.values.append(self.termination_node.value)

            if self.termination_node.release_chain is not None:
                print('test')
                tn = self.termination_node
                self.stop_chain()
                for i in range(int(tn.release_chain.duration)):
                    run_chain([sn])
                    self.values.append(self.termination_node.value)

        else:
            self.stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(SAMPLE_RATE), output=True,
                                 frames_per_buffer=FRAME_SIZE, stream_callback=callback)

        self.started = True

        if save_values:
            return self.values

    def stop_chain(self):
        if self.started and self.duration >= 0:
            if self.time_elapsed < self.duration:
                return

        if (self.termination_node.release_chain is not None and self.termination_node.release_chain.duration >= 0.0
                and not self.terminating):
            self.terminating = True

            self.termination_node.release_chain.source_node.register_upstream(self.termination_node)
            duration = self.termination_node.release_chain.duration
            self.termination_node = self.termination_node.release_chain.termination_node

            self.old_duration = self.duration
            self.duration = (self.duration if self.duration > 0 else 0) + duration

            return

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.termination_node = self.old_termination_node

        if self.termination_node.release_chain is not None:
            self.termination_node.release_chain.source_node.unregister_upstream(self.termination_node)
            self.termination_node.release_chain.reset_chain()

        self.reset_chain()

    def reset_chain(self):
        self.started = False
        self.terminating = False
        self.time_elapsed = 0.0
        self.duration = self.old_duration
        self.source_node.reset_chain()

    def visualize_chain(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        def node_id(node):
            return type(node).__name__ + '\n' + str(id(node)) + '\n' + node.get_display_properties()

        graph = pgv.AGraph(strict=False, directed=True)
        nodes = [self.source_node]
        while len(nodes) > 0:
            new_nodes = []
            for n in nodes:
                graph.add_node(node_id(n))
                for nd in n.downstream:
                    graph.add_edge(node_id(n), node_id(nd))
                new_nodes.extend(n.downstream)
            if nodes[0] == self.termination_node:
                if self.termination_node.release_chain is not None:
                    new_nodes = [self.termination_node.release_chain.source_node]
                    graph.add_edge(
                        node_id(self.termination_node),
                        node_id(self.termination_node.release_chain.source_node)
                    )
                    e = graph.get_edge(
                        node_id(self.termination_node),
                        node_id(self.termination_node.release_chain.source_node)
                    )
                    e.attr['color'] = 'turquoise'
            nodes = list(set(new_nodes))
        graph.layout(prog='dot')
        graph.draw('temp.png')
        img = mpimg.imread('temp.png')
        plt.imshow(img, interpolation='bicubic')
        plt.show()

    def chain_playviz(self, new_duration):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        old_duration = self.duration
        self.duration = new_duration
        values = self.play_chain(save_values=True)
        self.duration = old_duration

        plt.plot(range(len(values)), values, 'b-')
        plt.show()

    def tick(self):
        if self.started and self.duration >= 0:
            self.time_elapsed += 1.0/64.0 * SAMPLE_RATE
            if self.time_elapsed > self.duration:
                self.stop_chain()

    def set_duration(self, duration):
        self.old_duration = duration  # Not used the way it would imply here
        self.duration = duration
        return self

    @property
    def value(self):
        return self.termination_node.value

    @staticmethod
    def build_linear(*nodes):
        old_node = None

        for n in nodes:
            if old_node is not None:
                n.register_upstream(old_node)
            old_node = n

        return Chain(nodes[0], nodes[-1])


forth_decay = Chain.build_linear(
    LinearDecayNode(duration=BEAT_4TH),
    ChainTerminationNode()
).set_duration(BEAT_4TH)


eighth_decay = Chain.build_linear(
    LinearDecayNode(duration=BEAT_8TH),
    ChainTerminationNode()
).set_duration(BEAT_8TH)

whole_decay = Chain.build_linear(
    LinearDecayNode(duration=BEAT_WHOLE),
    ChainTerminationNode()
).set_duration(BEAT_WHOLE)


button_7 = Parameter(7)
button_7_start = ChainStartNode(button_7)
button_7_source1 = SineNode(frequency=Parameter(49.99)).register_upstream(button_7_start)
button_7_source2 = SineNode(frequency=Parameter(97.99)).register_upstream(button_7_start)
button_7_source3 = SawtoothNode(frequency=Parameter(146.83)).register_upstream(button_7_start)
button_7_out = ChainTerminationNode(release_chain=eighth_decay).register_upstream(button_7_source1)\
    .register_upstream(button_7_source2)\
    .register_upstream(button_7_source3)
button_7_chain = Chain(button_7_start, button_7_out)

button_6 = Parameter(6)
button_6_start = ChainStartNode(button_6)
# button_6_source = SineNode(frequency=Parameter(123.47)).register_upstream(button_6_start)
button_6_source = SineNode(frequency=Parameter(8), frequency_offset=110.0, frequency_multiplier=600.0)\
    .register_upstream(button_6_start)
button_6_attack = LinearAttackNode(duration=BEAT_WHOLE).register_upstream(button_6_source)
button_6_out = ChainTerminationNode(release_chain=whole_decay).register_upstream(button_6_attack)
button_6_chain = Chain(button_6_start, button_6_out)

button_5 = Parameter(5)
button_5_start = ChainStartNode(button_5)
button_5_source1 = SineNode(frequency=Parameter(73.4), translate=0.1).register_upstream(button_5_start)
button_5_source2 = TriangleNode(frequency=Parameter(73.4), translate=0.1).register_upstream(button_5_start)
button_5_source3 = TriangleNode(frequency=Parameter(36.7)).register_upstream(button_5_start)
button_5_out = ChainTerminationNode().register_upstream(button_5_source1)\
    .register_upstream(button_5_source2)\
    .register_upstream(button_5_source3)
button_5_chain = Chain(button_5_start, button_5_out)

# button_7_chain.chain_playviz(BEAT_WHOLE)

button_4 = Parameter(4)
kick_drum_start = ChainStartNode(button_4)
kick_drum_source = KickDrumNode().register_upstream(kick_drum_start)
kick_drum_out = ChainTerminationNode().register_upstream(kick_drum_source)
kick_drum_chain = Chain(kick_drum_start, kick_drum_out)

button_3 = Parameter(3)
hi_hat_start = ChainStartNode(button_3)
hi_hat_source = HiHatNode(pass_filter=0.3).register_upstream(hi_hat_start)
hi_hat_out = ChainTerminationNode().register_upstream(hi_hat_source)
hi_hat_chain = Chain(hi_hat_start, hi_hat_out)


button_2 = Parameter(2)

button_1 = Parameter(1)


def event_handler(sl, qt):
    global global_watchers
    global inputs
    inputs = sl
    while not qt.value:
        for w in global_watchers:
            w.tick()
        time.sleep(1.0/64.0)


aliases = {
    'Chain': ChainStartNode,
    'Sine': SineNode,
    'Triangle': TriangleNode,
    'Square': SquareNode,
    'Sawtooth': SawtoothNode
}


t = multiprocessing.Process(target=event_handler, args=(shared_list, quit_threads))
t2 = multiprocessing.Process(target=input_monitor, args=(shared_list, quit_threads))
t3 = multiprocessing.Process()
t.start()
t2.start()
t3.start()

command = ""

show_re = "show\s+(?P<chain>\w+)"
wave_re = "wave\s+(?P<dur>[0-9\.]+)\s+(?P<chain>\w+)"

while command not in ["quit", "exit"]:
    command = input("patch-cable > ")

    if command in ["quit", "exit"]:
        continue
    elif re.match(show_re, command):
        m = re.match(show_re, command).groupdict()
        eval('{}.visualize_chain()'.format(m['chain']))
    elif re.match(wave_re, command):
        m = re.match(wave_re, command).groupdict()
        print('{}.chain_playviz({})'.format(m['chain'], float(m['dur']) * SAMPLE_RATE))
        eval('{}.chain_playviz({})'.format(m['chain'], float(m['dur']) * SAMPLE_RATE))

quit_threads.value = 1

t2.join()
t.join()
