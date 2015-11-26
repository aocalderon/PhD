SELECT
	oid, ST_MakeLine(point) AS trajectory
FROM
	(SELECT
		oid, ST_MakePoint(lat, lon) AS point
	FROM
		oldenburg
	ORDER BY 
		oid, otime) AS a
GROUP BY
	oid