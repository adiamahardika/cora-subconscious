from app import db

class UserRole(db.Model):
    __tablename__ = 'user_roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "users": [
                {"id": user.id, "username": user.username, "email": user.email}
                for user in self.users
            ],
        }
