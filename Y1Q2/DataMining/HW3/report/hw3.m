function hw3()
	im=imread('pools3b.tif');
	im=im(:,:,1:3);
	image(im)
	[x,y]=ginput(50)
	dlmwrite('coords.txt',horzcat(x,y))
	d = function1(x,y);
	display(d)
	dlmwrite('newvalpools2.txt',d)

	function[data]=function1(x,y)
		for i=1:numel(x)
			x1=x(i);
			y1=y(i);
			red(i)=im(round(y1),round(x1),1);
			green(i)=im(round(y1),round(x1),2);
			blue(i)=im(round(y1),round(x1),3);
			class(i)=1;
		end
		data=[red;green;blue;class];
		data=transpose(data);
	end
end
