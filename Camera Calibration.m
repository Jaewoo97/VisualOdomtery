clear; close all; clc;
%% Calibration image path
left_calib='calibration pattern\pattern_left.jpg';
right_calib='calibration pattern\pattern_right.jpg';

%% Camera Calibration
clc; clear all; close all;
% Parameters
% Vital params
n_pts=18;
%Initialization
input_img_coords=zeros(n_pts,3);
input_wrd_coords=zeros(n_pts,4);
% Choosing pts & assignment of global coordinates
if isfile('calibration pattern\left_calib_points.mat') 
    load 'calibration pattern\left_calib_points.mat'
else
    img=imread(left_calib);
    imshow(img)
    hold on;
    for i=1:n_pts
        [x,y]=ginput(1);
        plot(x,y,'wx');
        txt=['\leftarrow pt',num2str(i)];
        text(x,y,txt,'color','white','fontsize',14);
        input_img_coords(i,:)=[x,y,1];
        foo=str2num(input('Input x,y,z world coordinates (ex: 1 2 3): ','s'));
        input_wrd_coords(i,:)=[foo(1),foo(2),foo(3),1];
    end
end
% Normalization of coordinates
% Transpose the coords to multiply by the similarity transformation matrix
img_coords=input_img_coords';
wrd_coords=input_wrd_coords';
% normalize using similarity matrix
mean_img=mean(input_img_coords); % 3 columns
mean_wrd=mean(input_wrd_coords); % 4 columns
dist_sum_img=0;
dist_sum_wrd=0;
for i=1:n_pts
    dist_sum_img=dist_sum_img+norm(input_img_coords(i,:)-mean_img);
    dist_sum_wrd=dist_sum_wrd+norm(input_wrd_coords(i,:)-mean_wrd);
end
s_img=sqrt(2)/sqrt(dist_sum_img/n_pts); % scaling for image coords
s_wrd=sqrt(3)/sqrt(dist_sum_wrd/n_pts); % scaling for world coords
norm_matrix_img=[s_img 0 -mean_img(1);0 s_img -mean_img(2); 0 0 1];
norm_img_coords=norm_matrix_img*img_coords;
norm_matrix_wrd=[s_wrd 0 0 -mean_wrd(1);0 s_wrd 0 -mean_wrd(2); 0 0 s_wrd -mean_wrd(3);0 0 0 1];
norm_wrd_coords=norm_matrix_wrd*wrd_coords;
foo=norm_img_coords';
norm_img_coords=foo;
foo=norm_wrd_coords';
norm_wrd_coords=foo;

% Build A matrix from xi, Xi
A=zeros(n_pts*2,12);
for n=1:n_pts
    A((n-1)*2+1,:)=[0 0 0 0 -norm_wrd_coords(n,1) -norm_wrd_coords(n,2) -norm_wrd_coords(n,3) -1 ...
    norm_img_coords(n,2)*norm_wrd_coords(n,1) norm_img_coords(n,2)*norm_wrd_coords(n,2) norm_img_coords(n,2)*norm_wrd_coords(n,3) norm_img_coords(n,2)];
    A((n-1)*2+2,:)=[norm_wrd_coords(n,1) norm_wrd_coords(n,2) norm_wrd_coords(n,3) 1 0 0 0 0 ...
    -norm_img_coords(n,1)*norm_wrd_coords(n,1) -norm_img_coords(n,1)*norm_wrd_coords(n,2) -norm_img_coords(n,1)*norm_wrd_coords(n,3) -norm_img_coords(n,1)];
end
% Computation of projection matrix
[U,S,V]=svd(A);
p_norm=reshape(V(:,end),4,3)';
% Find intrinsic , extrinsic properties
p=inv(norm_matrix_img)*p_norm*norm_matrix_wrd;
[U,S,V]=svd(p);
C=V(:,end);
Co_left=C/C(4);
M=p(:,1:3);
[R_inv K_inv]=qr(inv(M));
R_left=inv(R_inv);
K_temp=inv(K_inv);
K_left=K_temp/K_temp(3,3); % Scaling K
t_left=-R_left*Co_left(1:3,:);

if isfile('calibration pattern\right_calib_points.mat') 
    load 'calibration pattern\right_calib_points.mat'
else
    img=imread(right_calib);
    imshow(img)
    hold on;
    for i=1:n_pts
        [x,y]=ginput(1);
        plot(x,y,'wx');
        txt=['\leftarrow pt',num2str(i)];
        text(x,y,txt,'color','white','fontsize',14);
        input_img_coords(i,:)=[x,y,1];
        foo=str2num(input('Input x,y,z world coordinates (ex: 1 2 3): ','s'));
        input_wrd_coords(i,:)=[foo(1),foo(2),foo(3),1];
        save('calibration pattern\right_calib_points.mat','input_img_coords','input_wrd_coords')
    end
end
% Normalization of coordinates
% Transpose the coords to multiply by the similarity transformation matrix
img_coords=input_img_coords';
wrd_coords=input_wrd_coords';
% normalize using similarity matrix
mean_img=mean(input_img_coords); % 3 columns
mean_wrd=mean(input_wrd_coords); % 4 columns
dist_sum_img=0;
dist_sum_wrd=0;
for i=1:n_pts
    dist_sum_img=dist_sum_img+norm(input_img_coords(i,:)-mean_img);
    dist_sum_wrd=dist_sum_wrd+norm(input_wrd_coords(i,:)-mean_wrd);
end
s_img=sqrt(2)/sqrt(dist_sum_img/n_pts); % scaling for image coords
s_wrd=sqrt(3)/sqrt(dist_sum_wrd/n_pts); % scaling for world coords
norm_matrix_img=[s_img 0 -mean_img(1);0 s_img -mean_img(2); 0 0 1];
norm_img_coords=norm_matrix_img*img_coords;
norm_matrix_wrd=[s_wrd 0 0 -mean_wrd(1);0 s_wrd 0 -mean_wrd(2); 0 0 s_wrd -mean_wrd(3);0 0 0 1];
norm_wrd_coords=norm_matrix_wrd*wrd_coords;
foo=norm_img_coords';
norm_img_coords=foo;
foo=norm_wrd_coords';
norm_wrd_coords=foo;

% Build A matrix from xi, Xi
A=zeros(n_pts*2,12);
for n=1:n_pts
    A((n-1)*2+1,:)=[0 0 0 0 -norm_wrd_coords(n,1) -norm_wrd_coords(n,2) -norm_wrd_coords(n,3) -1 ...
    norm_img_coords(n,2)*norm_wrd_coords(n,1) norm_img_coords(n,2)*norm_wrd_coords(n,2) norm_img_coords(n,2)*norm_wrd_coords(n,3) norm_img_coords(n,2)];
    A((n-1)*2+2,:)=[norm_wrd_coords(n,1) norm_wrd_coords(n,2) norm_wrd_coords(n,3) 1 0 0 0 0 ...
    -norm_img_coords(n,1)*norm_wrd_coords(n,1) -norm_img_coords(n,1)*norm_wrd_coords(n,2) -norm_img_coords(n,1)*norm_wrd_coords(n,3) -norm_img_coords(n,1)];
end
% Computation of projection matrix
[U,S,V]=svd(A);
p_norm=reshape(V(:,end),4,3)';
% Find intrinsic , extrinsic properties
p=inv(norm_matrix_img)*p_norm*norm_matrix_wrd;
[U,S,V]=svd(p);
C=V(:,end);
Co_right=C/C(4);
M=p(:,1:3);
[R_inv K_inv]=qr(inv(M));
R_right=inv(R_inv);
K_temp=inv(K_inv);
K_right=K_temp/K_temp(3,3); % Scaling K
t_right=-R_right*Co_right(1:3,:);
camera_K = (K_left + K_right)/2; % Average Camera calibration matrix
baseline = sqrt( (abs(Co_left(1))-abs(Co_right(1)))^2 + (abs(Co_left(2))-abs(Co_right(2)))^2);
focal_length = (abs(K_left(1,1)) + abs(K_left(2,2)) + abs(K_right(1,1)) + abs(K_right(2,2))) / 4.0;


