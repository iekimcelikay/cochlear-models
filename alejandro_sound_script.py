import numpy as np
import scipy.signal
import thorns
import scipy.io.wavfile


def createStimulus(est, fs):
    if est['filename'] != -1:
        sound = readStimulus(est, fs)
    elif est['type'] == 'PT':
        sound = createPureTone(est, fs)
    elif est['type'] == 'IRN':
        sound = createIRN(est, fs)
    elif est['type'] == 'HCT':
        sound = createHCT(est, fs)
    elif est['type'] == 'altHCT':
        sound = createALTHCT(est, fs)
    elif est['type'] == 'chord':
        sound = createChord(est, fs)
    elif est['type'] == 'noise':
        sound = createNoise(est, fs)
    elif est['type'] == 'noiseIRN':
        sound = createNoiseIRN(est, fs)
    elif est['type'] == 'IRNchord':
        sound = createIRNchord(est, fs)
    elif est['type'] == 'IRNchordSS':
        sound = createIRNchordSingleSample(est, fs)
    elif est['type'] == 'HCTchord':
        sound = createHCTchord(est, fs)
    elif est['type'] == 'tricky':
        sound = createTrickyInterval(est, fs)
    elif est['type'] == 'CT':
        sound = createClickTrain(est, fs)
    elif est['type'] == 'CTchord':
        sound = createCTchord(est, fs)
    elif est['type'] == 'altCT':
        sound = createAltClicks(est, fs)
    elif est['type'] == 'IRNseq':
        sound = createIRNseq(est, fs)
    elif est['type'] == 'FMsweep':
        sound = createFMsweep(est, fs)
    elif est['type'] == 'transTone':
        sound = createTransposedTone(est, fs)
    else:
        print()
        'WARNING: Stimulus type not recognised, using a pure tone...'
        sound = createPureTone(est, fs)

    if est['filename'] == -1:

        if (type(est['bandpass']) == np.ndarray) and (est['bandpass'][0] != 0):
            sound = filterStim(sound, fs, est['bandpass'])

        sound = thorns.waves.set_dbspl(sound, est['loudness'])
        sound = rampInOut(sound, fs)
        sound = np.concatenate((np.zeros((int(est['onset'] * fs),)),
                                sound,
                                np.zeros(int(est['tail'] * fs))))

        if est['save'] == 1:
            scipy.io.wavfile.write('./mochStim' + est['type'] + '.wav', fs, sound)

    return sound


def createPureTone(est, fs):
    dur = est['duration']
    x = np.linspace(1 / fs, dur, int(dur * fs));

    omega = 2 * np.pi * est['freq']
    sound = np.sin(omega * x)

    return (sound)


def createChord(est, fs):
    dur = est['duration']
    x = np.linspace(0, dur, int(dur * fs));
    sound = np.zeros(int(est['duration'] * fs))

    for k in est['notes']:
        f0 = computeNoteFreq(est['freq'], k, est['tuning'])

        omega = 2 * np.pi * f0
        tone = np.sin(omega * x)

        sound = sound + tone

    return (sound)


def createHCT(est, fs):
    dur = est['duration']
    f0 = est['freq']
    sound = np.zeros(int(dur * fs))
    x = np.linspace(0, dur, int(dur * fs));

    for k in est['harms']:
        omega = 2 * np.pi * f0 * (k + 1)
        h = np.sin(omega * x)
        sound = sound + h * (est['harmFact'] ** k) / len(est['harms'])

    return (sound)


def createHCTchord(est, fs):
    dur = est['duration']

    sound = np.zeros(int(dur * fs))
    x = np.linspace(0, dur, int(dur * fs));

    for n in est['notes']:

        hct = np.zeros(int(dur * fs))
        f0 = computeNoteFreq(est['freq'], n, est['tuning'])

        for k in est['harms']:
            omega = 2 * np.pi * f0 * (k + 1)
            h = np.sin(omega * x)
            sound = sound + h * (est['harmFact'] ** k)

        sound = sound + hct

    return (sound)


def createIRN(est, fs):
    d = int(1. * fs / est['freq'])

    s = np.random.normal(0, 1, int(est['duration'] * fs + d * est['nOfIts']))
    sound = np.zeros(int(est['duration'] * fs))

    for k in range(est['nOfIts']):
        sound = sound + s[(d * k):(d * k + len(sound))]

    return (sound)


def createClickTrain(est, fs):
    sound = np.zeros(int(est['duration'] * fs))

    clicks = np.arange(1 / fs, est['duration'] * fs, fs / est['freq'])
    sound[clicks.astype(int)] = 1

    return (sound)


def createCTchord(est, fs):
    sound = np.zeros(int(est['duration'] * fs))

    for k in est['notes']:
        train = np.zeros(int(est['duration'] * fs))

        f0 = computeNoteFreq(est['freq'], k, est['tuning'])
        clicks = np.arange(fs / f0, est['duration'] * fs, fs / f0)

        train[clicks.astype(int)] = 1

        sound = sound + train

    return (sound)


def createNoise(est, fs):
    sound = np.random.normal(0, 1, int((est['duration'] * fs)))

    return (sound)


def createNoiseIRN(est, fs):
    noise = np.random.normal(0, 1, int((est['noiseOff'] * fs)))

    d = round(1. * fs / est['freq'])
    s = np.random.normal(0, 1, int((est['duration'] * fs + d * est['nOfIts'])))

    IRN = np.zeros(int(est['duration'] * fs))
    for k in range(est['nOfIts']):
        IRN = IRN + s[int((d * k)):int((d * k + len(IRN)))]

    IRN = thorns.waves.set_dbspl(IRN, est['loudness'])
    noise = thorns.waves.set_dbspl(noise, est['loudness'])

    sound = np.concatenate((noise, IRN))

    return (sound)


def createIRNchordSingleSample(est, fs):
    # Uses the same noise sample for all notes of the chord

    maxd = round((1. * fs) / (est['freq'])) + 1
    s = np.random.normal(0, 1, int(est['duration'] * fs + maxd * est['nOfIts']))
    sound = np.zeros(int(est['duration'] * fs))

    for k in est['notes']:

        f0 = computeNoteFreq(est['freq'], k, est['tuning'])
        d = int((1. * fs) / f0)
        h = np.zeros(int(est['duration'] * fs))

        for l in range(est['nOfIts']):
            h = h + s[(d * l):(d * l + len(sound))]

        sound = sound + h

    return (sound)


def computeNoteFreq(f0, note, tuning):
    just = (1., 16. / 15, 9. / 8, 6. / 5, 5. / 4, 4. / 3, 25. / 18,
            3. / 2, 8. / 5, 5. / 3, 9. / 5, 15. / 8, 2.)
    pythagorean = (1., 256. / 243, 9. / 8, 32. / 27, 81. / 64, 4. / 3, 1024. / 729,
                   3. / 2, 128. / 81, 27. / 16, 16. / 9, 243. / 128, 2.)

    if tuning == 'just':
        r = just[note]
    elif tuning == 'pyth':
        r = pythagorean[note]
    else:
        r = 2. ** ((1. * note) / 12)
        if tuning != 'equal':
            print
            'Warning: intonation not recognised, using equal tempered'

    f1 = r * f0

    return f1


def createIRNchord(est, fs):
    # Samples different noises for each note of the chord
    maxd = round((1. * fs) / (est['freq'])) + 1
    sound = np.zeros(int(est['duration'] * fs))

    for k in est['notes']:

        s = np.random.normal(0, 1, int(est['duration'] * fs + maxd * est['nOfIts']))
        f0 = computeNoteFreq(est['freq'], k, est['tuning'])
        d = int((1. * fs) / f0)
        h = np.zeros(int(est['duration'] * fs))

        for l in range(est['nOfIts']):
            h = h + s[(d * l):(d * l + len(sound))]

        sound = sound + h

    return (sound)


def createTrickyInterval(est, fs):
    tramps = 0.01
    tn = 0.3
    ttrans = est['duration'] - 2 * tramps - 2 * tn

    duration = tramps + tn + ttrans + tn + tramps;
    N = duration * fs;

    maxd = round((1. * fs) / (est['freq'])) + 1
    s = np.random.normal(0, 1, int((duration * fs + maxd * est['nOfIts'])))
    tone = [np.zeros(int(duration * fs)), np.zeros(int(duration * fs))]

    for i, k in enumerate(est['notes']):

        d = round((1. * fs) / (est['freq'] * 2. ** (1. * k / 12)))
        h = np.zeros(int(est['duration'] * fs))

        for l in range(est['nOfIts']):
            h = h + s[int(d * l):int(d * l + duration * fs)]

        tone[i] = filterStim(h, fs, est['loudness'], bandpass=(100, 4000))

    onN = tramps * fs
    toN = tn * fs
    trN = ttrans * fs

    on = np.sin(0.5 * np.pi * np.arange(tramps * fs) / (tramps * fs)) ** 2
    off = on[::-1]
    trOn = np.sin(0.5 * np.pi * np.arange(ttrans * fs) / (ttrans * fs)) ** 2
    trOff = 1 - trOn;

    a1 = np.concatenate([on, np.ones(int(tn * fs)), trOff, \
                         np.zeros(int((tn + tramps) * fs))])
    a2 = np.concatenate([np.zeros(int((tn + tramps) * fs)), trOn, \
                         np.ones(int(tn * fs)), off])

    sound = a1 * tone[0] + a2 * tone[1]
    sound = rampInOut(sound, fs)

    return (sound)


def createALTHCT(est, fs):
    dur = est['duration']
    f0 = est['freq']
    sound = np.zeros(int(dur * fs))
    x = np.linspace(0, dur, int(dur * fs));

    for k in est['harms']:
        omega = 2 * np.pi * f0 * (k + 1)
        phi = (np.pi / 2) * (k % 2)
        h = np.sin(omega * x + phi)
        sound = sound + h

    return (sound)


def createAltClicks(est, fs):
    d1 = 0.004
    d2 = 0.006

    sound = np.zeros(int(est['duration'] * fs))
    clks1 = np.arange(int((d1 + d2) * fs), int(est['duration'] * fs), \
                      int((d1 + d2) * fs))
    clks2 = np.arange(int(d1 * fs), int(est['duration'] * fs), \
                      int((d1 + d2) * fs))
    sound[clks1.astype(int)] = 1
    sound[clks2.astype(int)] = 1

    return (sound)


def createIRNseq(est, fs):
    # Samples different noises for each note of the chord
    maxd = round((1. * fs) / (est['freq'])) + 1

    dur = est['duration'] * len(est['notes'])
    sound = np.zeros(int(dur * fs))

    for k in range(len(est['notes'])):

        s = np.random.normal(0, 1, \
                             (int(est['duration'] * fs + maxd * est['nOfIts'])))
        h = np.zeros(int(est['duration'] * fs))

        f0 = computeNoteFreq(est['freq'], est['notes'][k], est['tuning'])
        d = round((1. * fs) / f0)

        for l in range(est['nOfIts']):
            h = h + s[int(d * l):int(d * l + len(h))]

        if k == 1:
            h = rampInOut(h, fs, ramp=False, damp=True)
        elif k == len(est['notes']):
            h = rampInOut(h, fs, ramp=True, damp=False)
        else:
            h = rampInOut(h, fs, ramp=True, damp=True)

        h = np.concatenate((np.zeros(int(k * est['duration'] * fs)), h, \
                            np.zeros(int((len(est['notes']) - k - 1) * est['duration'] * fs))))
        sound = sound + h

    return (sound)


def createFMsweep(est, fs):
    tau = est['noiseOff']
    dur = est['duration'] + 2 * tau

    f0 = est['freq'] - 0.5 * est['shift']
    f1 = est['freq'] + 0.5 * est['shift']

    f = np.concatenate((f0 * np.ones(int(tau * fs)), \
                        np.linspace(f0, f1, int(est['duration'] * fs)), \
                        f1 * np.ones(int(tau * fs))))

    sound = np.sin(2 * np.pi * np.cumsum(f) / fs);
    return sound


def createTransposedTone(est, fs):
    dur = est['duration']
    f0 = est['freq']
    fc = 5 * est['freq']

    x = np.linspace(0, dur, int(dur * fs));
    carrier = np.sin(2 * np.pi * fc * x)
    modulator = np.sin(2 * np.pi * f0 * x)
    modulator[modulator < 0] = 0

    sound = carrier * modulator

    return (sound)


def rampInOut(sound, fs, ramp=True, damp=True, tau=0.0025):
    L = int(tau * fs)
    hw = np.hamming(2 * L)
    sound[:L] = sound[:L] * hw[:L]
    sound[-L:] = sound[-L:] * hw[L:]

    return sound


def readStimulus(est, fs):
    (readfs, signal) = scipy.io.wavfile.read(est['filename'])
    sound = np.array(signal, dtype=float)

    if (est['intv'].size) == 2:
        sound = sound[int(est['intv'][0] * readfs):int(est['intv'][1] * readfs)]

    if est['loudness'] != -1:
        sound = thorns.waves.set_dbspl(sound, est['loudness'])

    sound = rampInOut(sound, fs)

    if not ('off' in est.keys()):
        est['onset'] = 0
    if not ('tail' in est.keys()):
        est['tail'] = 0

    sound = np.concatenate((np.zeros((int(est['onset'] * readfs),)),
                            sound,
                            np.zeros(int(est['tail'] * readfs))))

    sound = thorns.waves.resample(sound, readfs, fs)

    return (sound);


def filterStim(sound, fs, bandpass=(125, 2000)):
    fNyq = (bandpass[0] / (2. * fs), bandpass[1] / (2. * fs))
    b, a = scipy.signal.butter(2, fNyq, 'bandpass', analog=False)
    sound = scipy.signal.filtfilt(b, a, sound)

    return (sound)
