clc; clear;
data = load('data/HICO.mat');
HICO_original = data.HICO_original;
HICO_noisy = data.HICO_noisy;
hico_wl = data.hico_wl;
seawater_Rs = data.seawater_Rs;

size(HICO_original)
size(HICO_noisy)
size(hico_wl)
size(seawater_Rs)

% To convert a hyperspectral image cube I to matrix form X:
I = HICO_original;
[H,W,L] = size(I);
X = reshape(I, [H*W,L]);
X = X';

% To convert a matrix X back into a hyperspectral image cube:
I = reshape(X', [H,W,L]);

% Plot a single spectral band
imagesc(I(:,:,30));
axis('image');

% Note that quite a few libraries assume a matrix layout where
% each row is a spectral vector, rather than each column as in
% equation 2 of the assignment text. Read the documentation of
% those libraries carefully.
