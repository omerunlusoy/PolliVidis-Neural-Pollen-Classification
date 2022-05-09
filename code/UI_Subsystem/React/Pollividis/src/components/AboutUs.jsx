import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React from "react";

//
const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(0),
      padding: 20
  },
    card: {
        width: 300,
        height: 300,

    },
    media: {
        height: 200,
        width: 200,

    },

}));

const AboutUs = () => {
  const classes = useStyles();
  return (
      <div>
          <Navbar />
          <Typography style={{marginBottom:10, marginLeft:35, marginTop:100}}variant="h3" component="p">
              About Us
          </Typography>
          <Typography style={{marginBottom:20, marginLeft:35}} variant="h6" color="textSecondary" component="p">
              PolliVidis is developed by five Computer Science Students for their Senior Project.
          </Typography>
          <Grid container style={{marginLeft: 100}}>
              <Grid item xs={3} style={{marginRight: 20}} >
                  <Card className={classes.card}>
                      <CardActionArea onClick={()=> window.open("https://www.linkedin.com/in/elif-gamze-güliter/", "_blank")}>
                          <CardMedia className={classes.media} style={{marginLeft:50, marginTop:50}}   image="/p12_1.jpg"  />
                          <CardContent>
                              <Typography variant="h5" component="p" align="center">
                                  Elif Gamze Güliter
                              </Typography>
                          </CardContent>
                      </CardActionArea>
                  </Card>
              </Grid>
              <Grid item xs={3} style={{marginRight: 20}} >
                  <Card className={classes.card}>
                      <CardActionArea onClick={()=> window.open("https://www.linkedin.com/in/irem-tekin/", "_blank")}>
                          <CardMedia className={classes.media}  style={{marginLeft:50, marginTop:50}} image="/p12_2.jpg"  />
                          <CardContent>
                              <Typography variant="h5"  component="p" align="center">
                                  İrem Tekin
                              </Typography>
                          </CardContent>
                      </CardActionArea>
                  </Card>
              </Grid>
              <Grid item xs={3} style={{marginRight: 20}}>
                  <Card className={classes.card}>
                      <CardActionArea onClick={()=> window.open("https://www.linkedin.com/in/ece-ünal-a81b60116/", "_blank")}>
                          <CardMedia className={classes.media} style={{marginLeft:50, marginTop:50}}  image="/p12_3.jpg"  />
                          <CardContent>
                              <Typography variant="h5" component="p" align="center">
                                  Ece Ünal
                              </Typography>
                          </CardContent>
                      </CardActionArea>
                  </Card>
              </Grid>

              <Grid item xs={4} style={{marginLeft: 150, marginTop:30}} >
                  <Card className={classes.card}>
                      <CardActionArea onClick={()=> window.open("https://www.linkedin.com/in/omerunlusoy/", "_blank")}>
                          <CardMedia className={classes.media} style={{marginLeft:50, marginTop:50}}   image="/p12_4.jpg"  />
                          <CardContent>
                              <Typography variant="h5" component="p" align="center">
                                  Ömer Ünlüsoy
                              </Typography>
                          </CardContent>
                      </CardActionArea>
                  </Card>
              </Grid>
              <Grid item xs={4} style={{marginRight: 20, marginTop:30}}>
                  <Card className={classes.card}>
                      <CardActionArea onClick={()=> window.open("https://www.linkedin.com/in/umut-ada-yürüten-aa17051a0/", "_blank")}>
                          <CardMedia className={classes.media} style={{marginLeft:50, marginTop:50}}  image="/p12_5.jpg"  />
                          <CardContent>
                              <Typography variant="h5" component="p" align="center">
                                  Umut Ada Yürüten
                              </Typography>
                          </CardContent>
                      </CardActionArea>
                  </Card>
              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
