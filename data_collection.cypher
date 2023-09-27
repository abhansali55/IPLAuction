//VisualizeLot
MATCH (lot:Competition {compStructureTerm: "Auction lot"})
WITH lot LIMIT 1
MATCH p1 = (lot)-[:HAS_EVENT]->(bid:Event)
OPTIONAL MATCH p2 = (lot)<-[:INVOLVED_IN]-(:Participant)-[:HAS_ROLE]->(:Role)
OPTIONAL MATCH p6 = (lot)<-[:HAS_SUBSTRUC*1..4]-(:Competition)
OPTIONAL MATCH p7 = (lot)<-[:INVOLVED_IN]-(:Participant)<-[:HAS_PARTICIPANT*2]-(:Athlete)
OPTIONAL MATCH p3 = (bid)-[:FOLLOWS]->(:Event)
OPTIONAL MATCH p4 = (bid)-[:TYPE_OF]->(:Action)
OPTIONAL MATCH p5 = (bid)<-[:INVOLVED_IN]-(:SportSquad)<-[:HAS_SQUAD*2]-(:SportTeam)
RETURN p1,p2,p3,p4,p5,p6
-----------------------------------------------------------------------------------------------------------------------------------------
//PlayerBids
MATCH (bid:Event)-[r:TYPE_OF]->(a:Action)
WHERE a.actID = 3 OR a.actID = 4
WITH bid, r
MATCH p1 = (lot:Competition {compStructureTerm: "Auction lot"})-[:HAS_EVENT]->(bid)
WITH lot, COUNT(bid) AS numBids, max(r.amountCr) AS price
WHERE numBids >= 4
WITH lot, numBids, price
MATCH (ath:Athlete)-[:HAS_PARTICIPANT*2]->(partInLot:Participant)-[:INVOLVED_IN]->(lot)<-
						[:HAS_SUBSTRUC*1..4]-(season:Competition {compStructureTerm:"Season"})
OPTIONAL MATCH (partInLot)-[:HAS_ROLE]->(position:Role)
RETURN ath.entID AS playerID, ath.nameLastF AS playerName, season.startDate.year AS year, 
       position.termKRWebsite AS playerRole, numBids, price
ORDER BY playerID, year, playerRole, numBids
-----------------------------------------------------------------------------------------------------------------------------------------
//PlayerBidSequence
MATCH (ath:Athlete{entID:60703})-[:HAS_PARTICIPANT*2]->(partInLot:Participant)-[:INVOLVED_IN]->
(lot:Competition {compStructureTerm: "Auction lot"})<-[:HAS_SUBSTRUC*1..4]-(season:Competition {compStructureTerm:"Season"})
OPTIONAL MATCH (partInLot)-[:HAS_ROLE]->(position:Role)
WITH DISTINCT season, lot, ath, position
MATCH p1 = (lot)-[:HAS_EVENT]->(bid)WHERE (exists ( (bid)-[:TYPE_OF]->(:Action {actID:3})) OR 
					   exists( (bid)-[:TYPE_OF]->(:Action {actID:4}) ))  AND NOT 
				           exists ( (bid)-[:TYPE_OF]->(:Action {actID:2}) ) // Filter out lots with only a base price bid
WITH DISTINCT season, lot, ath, position,count(bid) AS numBids,collect(bid) AS BidsWHERE numBids >= 4
UNWIND Bids as bidMATCH (bid)<-[:INVOLVED_IN]-(:SportSquad)<-[:HAS_SQUAD*2]-(team:SportTeam)
RETURN DISTINCT bid.seqOrder AS bidNum,ath.entID AS playerID, ath.nameLastF AS playerName,season.startDate.year AS year,
		position.termKRWebsite AS playerRole,team.termKRWebsiteLong AS teamName
ORDER BY playerID, year, playerRole, bidNum
-----------------------------------------------------------------------------------------------------------------------------------------
//TeamSlotBidAvgPerYear
CALL {
	MATCH (bid:Event)-[r:TYPE_OF]->(a:Action)
	WHERE a.actID = 3 OR a.actID = 4
	WITH bid
	MATCH (lot:Competition {compStructureTerm: "Auction lot"})-[:HAS_EVENT]->(bid)<-[:INVOLVED_IN]-
					(:SportSquad)<-[:HAS_SQUAD*2]-(team:SportTeam)
	WHERE team.termKRWebsite IS NOT NULL AND (team.termKRWebsite <> "Gujarat Titans" 
						  AND team.termKRWebsite <> "Super Giants") AND bid.seqOrder > 0
	RETURN bid.seqOrder As eachBid, team.termKRWebsiteLong AS teamName, ROUND(COUNT(bid)/6.0,2) AS numBidsPerYear
	ORDER BY eachBid, teamName
								   
	UNION ALL
								   
	MATCH (bid:Event)-[r:TYPE_OF]->(a:Action)
	WHERE a.actID = 3 OR a.actID = 4
	WITH bid
	MATCH (lot:Competition {compStructureTerm: "Auction lot"})-[:HAS_EVENT]->(bid)<-[:INVOLVED_IN]-
					(:SportSquad)<-[:HAS_SQUAD*2]-(team:SportTeam)
	WHERE team.termKRWebsite IS NOT NULL AND (team.termKRWebsite = "Gujarat Titans" OR 
						  team.termKRWebsite = "Super Giants") AND bid.seqOrder > 0
	RETURN bid.seqOrder AS eachBid, team.termKRWebsiteLong AS teamName, ROUND(COUNT(bid)/2.0,2) AS numBidsPerYear
}
RETURN eachBid, teamName, numBidsPerYear
ORDER BY eachBid, teamName
