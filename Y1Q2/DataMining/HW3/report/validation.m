function validation()
	validateset=dlmread('valpools.csv');
	truegrid=dlmread('newpools_indeed.txt');
	truegridx=truegrid(:,1);
	truegridy=truegrid(:,2);
	x = modulo(validateset(:,1));
	y = modulo(validateset(:,2));
	validationgrids=horzcat(x,y);
	poolmap=containers.Map();

	for i=1:numel(x)
		flag=1;
		for j=1:numel(truegridx)
			if(x(i)==truegridx(j) && y(i)==truegridy(j))
				key=sprintf('%d,%d',x(i), y(i));
				poolmap(key)=1;
				flag=0;
				break;
			end
		end
		if(flag==1)
			key=sprintf('%d,%d',x(i), y(i));
			poolmap(key)=0;
		end
	end
	keys(poolmap)
	values(poolmap)

	function[x]=modulo(x)
		for i=1:numel(x)
			x(i)=x(i)-mod(x(i),50);
		end
	end
end
