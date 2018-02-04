#!/usr/bin/env python3

import pyaudio
import math
import numpy as np
import time
import itertools
import random
import functools


SAMPLE_RATE = 19200.0
VOLUME = 0.5

BEAT_32ND = SAMPLE_RATE / 32.0
BEAT_16TH = SAMPLE_RATE / 16.0
BEAT_8TH = SAMPLE_RATE / 8.0
BEAT_4TH = SAMPLE_RATE / 4.0
BEAT_HALF = SAMPLE_RATE / 2.0
BEAT_WHOLE = SAMPLE_RATE

TWO_PI = 2.0 * math.pi

p = pyaudio.PyAudio()

global_steppers = []
global_step = 0


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

    def attach_downstream(self, ds):
        self.downstream.append(ds)

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


class SourceNode(Node):
    def __init__(self, use_global_steps=False):
        global global_steppers

        super(SourceNode, self).__init__()
        self._x = 0
        self.use_global_steps = use_global_steps

        if self.use_global_steps:
            global_steppers.append(self)

    def step(self):
        self.value = self.function(global_step)


class OutputNode(Node):
    def __init__(self):
        super(OutputNode, self).__init__()


class RandomNoiseNode(SourceNode):
    def noise_fn(self, _):
        return self.translate + random.random() * self.amplitude

    def __init__(self, use_global_steps=False, translate=0.0, amplitude=1.0):
        super(RandomNoiseNode, self).__init__(use_global_steps)
        self.translate = translate
        self.amplitude = amplitude
        self.function = self.noise_fn


class SineNode(SourceNode):
    def sin(self, x):
        return self.translate + self.amplitude * math.sin(TWO_PI * (x / SAMPLE_RATE) * self.frequency)

    def __init__(self, frequency=440.0, use_global_steps=False, translate=0.0, amplitude=1.0):
        super(SineNode, self).__init__(use_global_steps)
        self.frequency = frequency
        self.function = self.sin
        self.translate = translate
        self.amplitude = amplitude


class SquareNode(SourceNode):
    def square(self, x):
        return math.copysign(1, math.sin(TWO_PI * (x / SAMPLE_RATE) * self.frequency))

    def __init__(self, frequency=440.0, use_global_steps=False):
        super(SquareNode, self).__init__(use_global_steps)
        self.frequency = frequency
        self.function = self.square


class SawtoothNode(SourceNode):
    def sawtooth(self, x):
        return self.amplitude * (-2.0 / math.pi * math.atan(1.0/math.tan(x * math.pi / (SAMPLE_RATE / self.frequency))))

    def __init__(self, frequency=440.0, use_global_steps=False, amplitude=0.5):
        super(SawtoothNode, self).__init__(use_global_steps)
        self.frequency = frequency
        self.function = self.sawtooth
        self.amplitude = amplitude


class BeatNode(SourceNode):
    def beat_fn(self, x):
        return self.translate + (self.amplitude if x % self.period_length <= self.beat_length else 0)

    def __init__(
            self,
            use_global_steps=False,

            translate=0.0,
            amplitude=1.0,

            beat_length=BEAT_4TH,
            gap_length=BEAT_4TH
    ):
        super(BeatNode, self).__init__(use_global_steps)
        self.translate = translate
        self.amplitude = amplitude
        self.beat_length = beat_length
        self.gap_length = gap_length
        self.period_length = self.beat_length + self.gap_length
        self.function = self.beat_fn


class FilterNode(Node):
    def filter_fn(self, x):
        return x * self.filter_param.value

    def __init__(self, filter_param):
        super(FilterNode, self).__init__()
        self.filter_param = filter_param

        self.function = self.filter_fn


cached_repeat = range(1024)


class Chain:
    def __init__(self, source_node, output_node):
        self.source_node = source_node
        self.output_node = output_node

    def play_chain(self, play_time=3.0):
        def callback(_in_data, _frame_count, _time_info, _status):
            global global_step

            output_samples = []

            for _ in cached_repeat:
                self.source_node.run_chain()

                global_step += 1

                for n in global_steppers:
                    n.run_chain()

                output_samples.append(self.output_node.value)

            return VOLUME * np.array(output_samples).astype(np.float32), pyaudio.paContinue

        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(SAMPLE_RATE), output=True,
                        stream_callback=callback)

        stream.start_stream()

        secs = play_time
        while stream.is_active() and secs > 0.2:
            secs -= 0.2
            time.sleep(0.2)

        stream.stop_stream()

        stream.close()

    @property
    def value(self):
        return self.output_node.value

    @staticmethod
    def build_linear(*nodes):
        old_node = None

        for n in nodes:
            if old_node is not None:
                n.register_upstream(old_node)
            old_node = n

        return Chain(nodes[0], nodes[-1])


class Parameter:
    PARAM_CONSTANT = 'PARAM_CONSTANT'
    PARAM_CHAIN = 'PARAM_CHAIN'
    PARAM_INPUT = 'PARAM_INPUT'

    def __init__(self, param_type, param_value):
        self.param_type = param_type
        self.param_value = param_value

    @property
    def value(self):
        if self.param_type == Parameter.PARAM_CONSTANT:
            return self.param_value
        if self.param_type == Parameter.PARAM_CHAIN:
            return self.param_value.value
        if self.param_type == Parameter.PARAM_INPUT:
            return 0  # TODO
        else:
            return 0


test_filter_param = Parameter(param_type=Parameter.PARAM_CHAIN, param_value=Chain.build_linear(
    BeatNode(use_global_steps=True, beat_length=BEAT_4TH, gap_length=BEAT_4TH*3),
    OutputNode()
))

test_dummy = SourceNode()
test_source = RandomNoiseNode(amplitude=0.5)\
    .register_upstream(test_dummy)
test_source_2 = SquareNode(220.0)\
    .register_upstream(test_dummy)
test_filter = FilterNode(test_filter_param)\
    .register_upstream(test_source)
test_out = OutputNode() \
    .register_upstream(test_filter) \
    .register_upstream(test_source_2)

test_chain = Chain(test_dummy, test_out)

inputs = []
variables = {}

test_chain.play_chain()

p.terminate()

# TODO: Quantizing?
