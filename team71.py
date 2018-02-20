import copy
import random
import time

class Player71:
    def __init__(self):
        self.useDynaDepth = True
        self.startTime = 0
        self.time_limit = 14.5

        self.validBlocks= [[0 for i in range(4)] for j in range(4)]
        self.validBlocks[0][0]=((0,0),)
        self.validBlocks[0][1]=((0,1),)
        self.validBlocks[0][2]=((0,2),)
        self.validBlocks[0][3]=((0,3),)
        self.validBlocks[1][0]=((1,0),)
        self.validBlocks[1][1]=((1,1),)
        self.validBlocks[1][2]=((1,2),)
        self.validBlocks[1][3]=((1,3),)
        self.validBlocks[2][0]=((2,0),)
        self.validBlocks[2][1]=((2,1),)
        self.validBlocks[2][2]=((2,2),)
        self.validBlocks[2][3]=((2,3),)
        self.validBlocks[3][0]=((3,0),)
        self.validBlocks[3][1]=((3,1),)
        self.validBlocks[3][2]=((3,2),)
        self.validBlocks[3][3]=((3,3),)
        #self.allList = ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3))
        self.allList = []
        for i in range(4):
            for j in range(4):
                self.allList.append((i,j))
        self.heuristicDict = {}

        #self.getBlockScore([[0]*4 for i in range(4)])

    def checkAllowedBlocks(self, prevMove, BlockStatus):
        if prevMove[0] < 0 and prevMove[1] < 0:
            return self.allList
        allowedBlocks = self.validBlocks[prevMove[0]%4][prevMove[1]%4]
        finalAllowedBlocks = []
        for i in allowedBlocks:
            if BlockStatus[i[0]][i[1]] == 0:
                finalAllowedBlocks.append(i)
        if len(finalAllowedBlocks) == 0:
            for i in self.allList:
                if BlockStatus[i[0]][i[1]] == 0:
                    finalAllowedBlocks.append(i)
        return finalAllowedBlocks

    def checkAllowedMarkers(self,block):
        allowed=[]
        for i in range(4):
            for j in range(4):
                if block[i][j] == 0:
                    allowed.append((i, j))
        return allowed

    def getAllowedMoves(self, currentBoard, currentBlockStatus, prevMove):
        moveList=[]
        for allowedBlock in self.checkAllowedBlocks(prevMove, currentBlockStatus):
            moveList += [(4*allowedBlock[0]+move[0], 4*allowedBlock[1]+move[1]) for move in self.checkAllowedMarkers(currentBoard[allowedBlock[0]][allowedBlock[1]])]
        return moveList

    def getBlockStatus(self,block):
        for i in range(4):
            if block[i][0]==block[i][1]==block[i][2]==block[i][3] and block[i][1] in (1,2):
                return block[i][1]
            if block[0][i]==block[1][i]==block[2][i]==block[3][i] and block[1][i] in (1,2):
                return block[1][i]

        if block[0][0]==block[1][1]==block[2][2]==block[3][3] and block[1][1] in (1,2):
            return block[1][1]
        if block[0][3]==block[2][1]==block[1][2]==block[3][0] and block[2][1] in (1,2):
            return block[2][1]

        if not len(self.checkAllowedMarkers(block)):
            return 3
        return 0

    def count(self, i, j, block):

        row_flag = 1
        col_flag = 1
        pdiag_flag = 1
        ndiag_flag = 1
        self.score = 0
        self.total = 0
        new_block = copy.deepcopy(block)
        #print new_block
        for row in range(4):
            if(new_block[i][row] == 2):
                row_flag = 0
        if(row_flag):
            self.total = 1
            for row in range(4):
                #print " * ", i, " ",row," ", new_block[i][row]
                if(new_block[i][row] == 1):
                    #print "1.\n"
                    self.total *= 3
            self.score += self.total

        for col in range(4):
            if(new_block[col][j] == 2):
                col_flag = 0
        if(col_flag):
            self.total = 1
            for col in range(4):
                if(new_block[col][j] == 1):
                    #print "2.\n"
                    self.total *= 3
            self.score += self.total

        if((i,j) in [(0,0),(1,1),(2,2),(3,3)]):
            for diag in range(4):
                if(new_block[diag][diag] == 2):
                    pdiag_flag = 0
            if(pdiag_flag):
                self.total = 1
                for diag in range(4):
                    if(new_block[diag][diag] == 1):
                        #print "3.\n"
                        self.total *= 3
                self.score += self.total

        if((i,j) in [(0,3),(1,2),(2,1),(3,0)]):
            for iter_i in range(4):
                if(new_block[iter_i][3-iter_i] == 2):
                    ndiag_flag = 0
            if(pdiag_flag):
                self.total = 1
                for iter_i in range(4):
                    if(new_block[iter_i][3-iter_i] == 1):
                        #print "4.\n"
                        self.total *= 3
                self.score += self.total

        return self.score

    def getBlockScore(self,block):
        block = tuple([tuple(block[i]) for i in range(4)])
        if block not in self.heuristicDict:
            blockStat = self.getBlockStatus(block)
            if blockStat == 1:
                self.heuristicDict[block] = 100
            elif blockStat == 2:
                self.heuristicDict[block] = 0 #Check whether to keep 0.0 or not
            elif blockStat == 3:
                self.heuristicDict[block] = 0
            else:
                best = -100000
                moves = self.checkAllowedMarkers(block)
                playBlock = [list(block[i]) for i in range(4)]
                opnplayBlock = [list(block[i]) for i in range(4)]
                for i in range(4):
                    for j in range(4):
                        if opnplayBlock[i][j]:
                            opnplayBlock[i][j] = 3 - opnplayBlock[i][j]

                for move in moves:
                    ans = 1.0 + 1*self.count(move[0],move[1],playBlock) - 0*self.count(move[0],move[1],opnplayBlock)
                    #Check whether to keep 0.99 or not

                    if ans > best:
                        wePlayList = []
                        best = ans
                        #print move[0],move[1],ans
                        wePlayList.append((move[0],move[1]))
                    elif ans == best:
                        wePlayList.append((move[0],move[1]))

                    self.heuristicDict[block] = best
        return self.heuristicDict[block]

    def lineScore(self, line, blockProb, revBlockProb, currentBlockStatus):
        #print line
        if 3 in [currentBlockStatus[x[0]][x[1]] for x in line]:
            return 0
        positiveScore = [blockProb[x[0]][x[1]] for x in line]
        negativeScore = [revBlockProb[x[0]][x[1]] for x in line]

        pos = 1
        neg = 1
        for i in positiveScore:
            pos = pos*i
        for i in negativeScore:
            neg = neg*i

        return pos-neg

    def getBoardScore(self, currentBoard, currentBlockStatus):
        terminalStat, terminalScore = self.terminalCheck(currentBoard, currentBlockStatus)
        if terminalStat:
            return terminalScore
        revCurrenBoard = copy.deepcopy(currentBoard)
        for row in range(4):
            for col in range(4):
                for i in range(4):
                    for j in range(4):
                        if revCurrenBoard[row][col][i][j]:
                            revCurrenBoard[row][col][i][j] = 3 - revCurrenBoard[row][col][i][j]

        blockProb = [[0]*4 for i in range(4)]
        revBlockProb = [[0]*4 for i in range(4)]
        for i in range(4):
            for j in range(4):
                blockProb[i][j] = self.getBlockScore(currentBoard[i][j])
                revBlockProb[i][j] = self.getBlockScore(revCurrenBoard[i][j])

        #print revBlockProb
        boardScore = []

        for i in range(4):
            line = [(i,j) for j in range(4)]
            boardScore.append(self.lineScore(line, blockProb, revBlockProb, currentBlockStatus))
            line = [(j,i) for j in range(4)]
            boardScore.append(self.lineScore(line, blockProb, revBlockProb, currentBlockStatus))


        line = [(i,i) for i in range(4)]
        boardScore.append(self.lineScore(line, blockProb, revBlockProb, currentBlockStatus))
        line = [(i,3-i) for i in range(4)]
        boardScore.append(self.lineScore(line, blockProb, revBlockProb, currentBlockStatus))
        #print boardScore

        # Condition for winning should be written here
        if 100000000 in boardScore:
            #print "found win", currentBoard
            return 100000000
        elif -100000000 in boardScore: # Check constant value
            return -100000000
        #print sum(boardScore)
        return sum(boardScore)


    def move(self,currentBoard,oldMove,flag):
        self.startTime = time.time()
        formattedBoard = [[[[0]*4 for i in range(4)] for j in range(4)] for j in range(4)]
        formattedBlockStatus = [[0]*4 for i in range(4)]
        copyBlock = [[0]*4 for i in range(4)]

        #print currentBoard.block_status, flag

        for i in range(16):
            for j in range(16):
                if currentBoard.board_status[i][j] == flag:
                    formattedBoard[i/4][j/4][i%4][j%4] = 1
                elif currentBoard.board_status[i][j] == '-':
                    formattedBoard[i/4][j/4][i%4][j%4] = 0
                else:
                    formattedBoard[i/4][j/4][i%4][j%4] = 2

        # Check this module with simulator
        for i in range(4):
            for j in range(4):
                if currentBoard.block_status[i][j] == flag:
                    formattedBlockStatus[i][j] = 1
                elif currentBoard.block_status[i][j] == '-':
                    formattedBlockStatus[i][j] = 0
                elif currentBoard.block_status[i][j] == 'd':
                    formattedBlockStatus[i][j] = 3
                else:
                    formattedBlockStatus[i][j] = 2

        #print formattedBlockStatus

        if oldMove[0] < 0 or oldMove[1] < 0:
            uselessScore, nextMove, retDepth = 0,(10,10),0
            depth = 0
        else:
            if self.useDynaDepth:
                for i in range(3,20):
                    #print i
                    depth = i
                    uselessScore,best_move, retDepth = self.alphaBetaPruning(formattedBoard,formattedBlockStatus,-100000000000000000,100000000000000000, True, oldMove, depth)
                    if (time.time() - self.startTime) < self.time_limit:
                        print i, best_move, uselessScore
                        nextMove = best_move
                    else:
                        break
            else:
                depth = 4
                uselessScore,nextMove, retDepth = self.alphaBetaPruning(formattedBoard,formattedBlockStatus,-100000000000000000,100000000000000000, True, oldMove, depth)
        #print "nextMove", nextMove, uselessScore
        #print time.time() - self.startTime
        try:
            return nextMove
        except:
            possibMoves = self.getAllowedMoves(currentBoard, currentBlockStatus, prevMove)
            random.shuffle(possibMoves)
            nextMove = possibMoves[0]
            return nextMove

    def terminalCheck(self, currentBoard, currentBlockStatus):
        terminalStat = self.getBlockStatus(currentBlockStatus)
        if terminalStat == 0:
            return (False,0)
        elif terminalStat == 1:
            return (True, 100000000)
        elif terminalStat == 2:
            return   (True, -100000000)
        else:
            blockCount = 0
            midCount = 0
            for i in range(4):
                for j in range(4):
                    if currentBlockStatus[i][j] in (1,2):
                        blockCount += 10*(3-2*currentBlockStatus[i][j]) #Check whether to keep 10 or 5

                    # Check whether to keep midCount or not

            return (True, blockCount)

    def alphaBetaPruning(self, currentBoard, currentBlockStatus, alpha, beta, flag, prevMove, depth):
        #print self.useDynaDepth
        tempBoard = copy.deepcopy(currentBoard)
        tempBlockStatus = copy.deepcopy(currentBlockStatus)
        terminalStat, terminalScore = self.terminalCheck(currentBoard, currentBlockStatus)
        #print "terminal",terminalStat
        if terminalStat:
            return terminalScore, (), 0

        if self.useDynaDepth:
            #print "Dyna",time.time(), self.startTime
            if (time.time() - self.startTime) > self.time_limit:
                return 0, (), 0

        if depth<=0:
            #print currentBlockStatus
            return self.getBoardScore(currentBoard, currentBlockStatus), (), 0

        possibMoves = self.getAllowedMoves(currentBoard, currentBlockStatus, prevMove)
        random.shuffle(possibMoves)
        bestMove = ()
        bestDepth = 100
        if flag:
            v = -1000000000
            for move in possibMoves:
                #implement the move
                tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 1
                tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])

                #print move
                childScore, childBest, childDepth = self.alphaBetaPruning(tempBoard, tempBlockStatus, alpha, beta, not flag, move, depth-1)
                if self.useDynaDepth:
                    if (time.time() - self.startTime) > self.time_limit:
                        return 0, (), 0
                #print childScore
                if childScore >= v:
                    if v < childScore or bestDepth > childDepth:
                        v = childScore
                        bestMove = move
                        bestDepth = childDepth
                alpha = max(alpha, v)

                #revert the implemented move
                tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 0
                tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])

                if alpha >= beta:
                    break

            return v, bestMove, bestDepth+1
        else:
            v = 1000000000
            for move in possibMoves:
                #implement the move
                tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 2
                tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])

                #print move
                childScore, childBest, childDepth = self.alphaBetaPruning(tempBoard, tempBlockStatus, alpha, beta, not flag, move, depth-1)
                if self.useDynaDepth:
                    if (time.time() - self.startTime) > self.time_limit:
                        return 0, (), 0
                #print childScore
                if childScore <= v:
                    if v > childScore or bestDepth > childDepth:
                        v = childScore
                        bestMove = move
                        bestDepth = childDepth
                beta = min(beta, v)

                #revert the implemented move
                tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 0
                tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])

                if alpha >= beta:
                    break

            return v, bestMove, bestDepth+1



'''
if __name__ == '__main__':
    obj = Player71()
    block = [
        [0,0,0,0],
        [2,2,2,0],
        [1,1,1,0],
        [0,0,0,0]
    ]
    ans = obj.getBlockScore(block)
    print ans
'''
