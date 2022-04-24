import {
  Button,
  Card,
  CardActionArea,
  CardActions,
  CardContent,
  CardMedia,
  makeStyles,
  Typography,
} from "@material-ui/core";

const useStyles = makeStyles((theme) => ({
  card: {
    width: 560,
    height:460,
    marginBottom: theme.spacing(5),
    ///marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
  },
  media: {
    height: 450,
    width: 550,
    [theme.breakpoints.down("sm")]: {
      height:450,
      width: 550
    },
  },
}));

const SampleImagePreviewCard = ({ img}) => {
  const classes = useStyles();
  return (
      <Card className={classes.card}>
        <CardActionArea>
          <CardMedia className={classes.media} image={img}  />
        </CardActionArea>
      </Card>
  );
};

export default SampleImagePreviewCard;
