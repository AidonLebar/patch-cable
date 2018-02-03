#!/usr/bin/env python3

import pyaudio
import math
import numpy as np


SAMPLE_RATE = 44100
VOLUME = 0.3

p = pyaudio.PyAudio()


class Node:
    def __init__(self):
        self.upstream = []
        self.downstream = []
        self.function = lambda x: x
        self.value = self.function(0)

    def register_upstream(self, up):
        self.upstream.append(up)

    def attach_downstream(self, ds):
        print(self.downstream)
        self.downstream.append(ds)
        ds.register_upstream(self)

    def reckon(self):
        return sum([up.value for up in self.upstream]) / len(self.upstream)

    def step(self):
        self.value = self.function(self.reckon())

    def run_chain(self):
        self.step()
        for ds in self.downstream:
            ds.step()


class SourceNode(Node):
    def __init__(self, frequency=440.0):
        super(SourceNode, self).__init__()
        self._x = 0
        self.frequency = frequency
        self.upstream = None  # No upstream allowed

    def step(self):
        self._x += 1
        self.value = self.function(self._x)


class OutputNode(Node):
    def __init__(self):
        super(OutputNode, self).__init__()
        self.downstream = None  # No downstream allowed


class SineNode(SourceNode):
    def sin(self, x):
        # print(math.sin(2.0 * math.pi * (x / float(SAMPLE_RATE)) * self.frequency))
        return math.sin(2.0 * math.pi * (x / float(SAMPLE_RATE)) * self.frequency)

    def __init__(self, frequency=440.0):
        super(SineNode, self).__init__(frequency)
        self.function = self.sin


class SquareNode(SourceNode):
    def square(self, x):
        # print(math.copysign(1, math.sin(2.0 * math.pi * (x / float(SAMPLE_RATE)) * self.frequency)))
        return math.copysign(1, math.sin(2.0 * math.pi * (x / float(SAMPLE_RATE)) * self.frequency))

    def __init__(self, frequency=440.0):
        super(SquareNode, self).__init__(frequency)
        self.function = self.square


def play_chain(source: SourceNode, output: OutputNode):
    output_samples = []
    for i in range(SAMPLE_RATE * 2):
        source.run_chain()
        output_samples.append(output.value)

    print(len(output_samples))

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True)
    stream.write(VOLUME * np.array(output_samples).astype(np.float32))
    print(output_samples[:10])
    stream.close()


variables = {}

test_source = SquareNode(100.0)
test_out = OutputNode()

test_source.attach_downstream(test_out)

play_chain(test_source, test_out)

p.terminate()

# TODO: Quantizing
