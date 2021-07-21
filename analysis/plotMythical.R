
require(ggplot2)
require(RColorBrewer)

# Giving this file a .txt instead of .csv extension so that git doesn't ignore
# it (we told it we don't want to commit/push .csv files, since most of ours
# are so big.)
mdf <- read.csv("mythicalDataFrame.txt")

p <- ggplot(mdf,
    aes(x=year, y=perc_polar, color=subreddit, group=subreddit)) +
    geom_line(aes(color=subreddit), size=1) +
    scale_color_brewer(palette="Paired") +
    scale_x_continuous(name="Year", breaks=c(2008:2021)) +
    ylim(c(-5,105)) +
    ylab("% of Threads Polarized") +
    labs(title="Estimated Polarization of Subreddits Over Time") +
    theme(plot.title=element_text(hjust=0.5))
ggsave("mythicalGraphBySubreddit.png", height=10/3, width=29/3)
