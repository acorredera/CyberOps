function [errorCode] = python_FFT_spectrum(path_audio_input,path_audio_output,fs, N, frameSize, frameStep)
    addpath(genpath('./AAdata/libs/MIRtoolbox1.7'))
    waveform = miraudio(path_audio_input, 'Sampling',fs,'Frame', frameSize, 's',frameStep, 's');
    spectrum = mirspectrum(waveform, 'Length',N, 'Min',0,'Max',fs/2,'dB');%'Min',20,'Max',8000,, 'Length',512; mirspectrum(...,'Power'): squares the energy.
    spectrum_data = mirgetdata(spectrum);
    [rows, cols] = size(spectrum_data);
    for i=1:(rows)
        fixed_spectrum_matrix(i,:) = spectrum_data(end-i+1,:);
    end
     save(path_audio_output, 'fixed_spectrum_matrix');
    errorCode = 0
end