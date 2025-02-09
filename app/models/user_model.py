from app import db

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    email = db.Column(db.String(100))
    role_id = db.Column(db.Integer, db.ForeignKey('user_roles.id'))
    
    role = db.relationship("UserRole", backref="users")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.to_dict() if self.role else None
        }
