function hw3()
im=imread('pools3.tif');
im=im(:,:,1:3);
image(im)
[x,y]=ginput(5)
d = function1(x,y);
display(d)

function[data]=function1(x,y)

for i=1:numel(x)
    x1=x(i);
    y1=y(i);
    red(i)=im(round(y1),round(x1),1);
    green(i)=im(round(y1),round(x1),2);
    blue(i)=im(round(y1),round(x1),3);
end

data=[red;green;blue];
data=transpose(data);
end
end