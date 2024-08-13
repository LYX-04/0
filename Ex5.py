#最长回文子串

##方法一：动态规划
class Solution:
    def longestPalindrome(self,s:str)->str:
        n=len(s)
        if n<2:
            return s

        max_len=1
        begin=0
        dp=[[False]*n for _ in range(n)]

        #初始化单字符都是回文
        for i in range(n):
            dp[i][i]=True

        #枚举子串长度
        for L in range(2,n+1):

            #枚举左边界，注意这里限制了右边界不会超出字符串长度
            #遍历所有的起始索引
            for i in range(n-L+1):

                #结束的位置
                j=i+L-1
                
                #判断起始和终止位置是否相等
                if s[i]!=s[j]:
                    dp[i][j]=False

                else:
                    if L==2:  #如果子串长度为2，直接判断两端字符是否相等
                        dp[i][j]=True 
                    else:
                        dp[i][j]=dp[i+1][j-1]

                #更新最长回文子串
                if dp[i][j] and L>max_len:
                    max_len=L
                    begin=i

        return s[begin:begin+max_len]

##方法二：中心扩展算法
class Solution:
    def expandAroundCenter(self,s,left,right):
        while left>=0 and right<len(S) and s[left]==s[right]:
            left-=1
            right+=1
        return left+1,right-1

    def longestPalidrome(self,s:str)->str:

        start,end=0,0

        for i in range (len(s)):
            
            #单字符中心
            left1,right1=self.expandAroundCenter(s,i,i)
            #双字符中心
            left2,right2=self.expandAroundCenter(s,i,i+1)

            if right1-left1>end-start:
                start,end=left1,right1

            if right2-left2>end-start:
                strat,end=left2,right2

        return s[start:end+1]





